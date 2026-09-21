"""Observed-pair DeepDEP training with epoch-level atomic restart checkpoints."""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch

from advanced_model_methods import ExpDeepDEPAdapted, initialize_he
from run_advanced_model_benchmark import sha256


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def initialize(encoder, fingerprint_n, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ExpDeepDEPAdapted(encoder, fingerprint_n).cuda()
    initialize_he(model.fingerprint_encoder)
    initialize_he(model.predictor)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9,.999), eps=1e-7)
    return model, optimizer


def paired_forward(model, expression, fingerprints, pairs):
    sample = model.expression_encoder(expression[pairs[:,0]])
    gene = model.fingerprint_encoder(fingerprints[pairs[:,1]])
    return model.predictor(torch.cat((sample,gene),dim=1)).squeeze(1)


def update(model, optimizer, x, y, fp, pairs):
    optimizer.zero_grad(set_to_none=True)
    prediction = paired_forward(model,x,fp,pairs)
    loss = (prediction-y[pairs[:,0],pairs[:,1]]).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('训练损失非有限值，停止且保留最近完整epoch断点')
    loss.backward()
    optimizer.step()
    return loss.detach()


def tensors(expression, outcomes, fingerprints):
    x=torch.as_tensor(expression,dtype=torch.float32,device='cuda')
    y=torch.as_tensor(outcomes,dtype=torch.float32,device='cuda')
    fp=torch.as_tensor(fingerprints,dtype=torch.float32,device='cuda')
    pairs=torch.nonzero(torch.isfinite(y),as_tuple=False)
    if len(pairs)==0 or not torch.isfinite(x).all() or not torch.isfinite(fp).all():
        raise ValueError('输入非有限或没有有效依赖标签')
    return x,y,fp,pairs


def predict(model, expression, fp):
    model.eval()
    result=[]
    with torch.no_grad():
        for batch in torch.as_tensor(expression,dtype=torch.float32,device='cuda').split(32):
            result.append(model.cartesian(batch,fp).cpu())
    model.train()
    return torch.cat(result)


def benchmark(encoder, expression, outcomes, fingerprints, seed):
    model,optimizer=initialize(encoder,fingerprints.shape[1],seed)
    x,y,fp,pairs=tensors(expression,outcomes,fingerprints)
    order=torch.randperm(len(pairs),device='cuda')
    batches=order.split(500)
    if len(batches)<220:
        raise ValueError('计时训练子集不足220个批次')
    for batch in batches[:20]:update(model,optimizer,x,y,fp,pairs[batch])
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    tick=time.monotonic()
    for batch in batches[20:220]:update(model,optimizer,x,y,fp,pairs[batch])
    torch.cuda.synchronize();seconds=time.monotonic()-tick
    return {'warmup_updates':20,'timed_updates':200,'seconds':seconds,
            'seconds_per_update':seconds/200,'pairs_per_batch':500,
            'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),
            'training_model_n':len(expression),'observed_training_pair_n':len(pairs)}


def fit(encoder,expression,outcomes,held_expression,fingerprints,seed,epochs,checkpoints,directory,label):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    spec={'seed':seed,'epochs':epochs,'checkpoints':list(checkpoints),
          'train_shape':list(expression.shape),'target_shape':list(outcomes.shape),'held_n':len(held_expression)}
    specification=directory/'spec.json'
    if specification.exists() and json.loads(specification.read_text())!=spec:
        raise ValueError(f'拟合断点设置变化：{directory}')
    atomic_json(specification,spec)
    done=directory/'done.json'
    if done.exists():
        metadata=json.loads(done.read_text())
        for name,digest in metadata['output_sha256'].items():
            if sha256(directory/name)!=digest:raise ValueError(f'拟合缓存损坏：{name}')
        with np.load(directory/'predictions.npz',allow_pickle=False) as archive:
            return {int(k):archive[k] for k in archive.files}
    model,optimizer=initialize(encoder,fingerprints.shape[1],seed)
    x,y,fp,pairs=tensors(expression,outcomes,fingerprints)
    state_path=directory/'checkpoint.pt';first_epoch=1;predictions={};history=[]
    if state_path.exists():
        state=torch.load(state_path,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        first_epoch=state['epoch']+1;predictions=state['predictions'];history=state['history']
        torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        print(f'    【续跑】{label}｜种子 {seed}｜从epoch {first_epoch}开始',flush=True)
    for epoch in range(first_epoch,epochs+1):
        tick=time.monotonic();total=torch.zeros((),device='cuda')
        order=torch.randperm(len(pairs),device='cuda')
        last_report=tick
        for step,batch in enumerate(order.split(500),1):
            total+=update(model,optimizer,x,y,fp,pairs[batch])*len(batch)
            if time.monotonic()-last_report>=30:
                print(f'    【训练】{label}｜种子 {seed}｜epoch {epoch}/{epochs}｜批次 {step}/{(len(pairs)+499)//500}',flush=True)
                last_report=time.monotonic()
        mse=float(total/len(pairs))
        if epoch in checkpoints:
            predictions[epoch]=predict(model,held_expression,fp)
        history.append({'epoch':epoch,'training_mse':mse,'elapsed_seconds':time.monotonic()-tick})
        state={'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
               'predictions':predictions,'history':history,'cpu_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state()}
        temporary=state_path.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(state_path)
        print(f'    【训练】{label}｜种子 {seed}｜epoch {epoch}/{epochs}｜训练MSE {mse:.6f}｜{time.monotonic()-tick:.1f}秒',flush=True)
    arrays={str(k):v.numpy() for k,v in predictions.items()}
    if set(map(int,arrays))!=set(checkpoints):raise ValueError('拟合完成但预测检查点不完整')
    temporary=directory/'predictions.tmp'
    with temporary.open('wb') as handle:np.savez_compressed(handle,**arrays)
    temporary.replace(directory/'predictions.npz')
    pd.DataFrame(history).to_csv(directory/'training.csv',index=False)
    atomic_json(done,{'spec':spec,'output_sha256':{name:sha256(directory/name) for name in ['predictions.npz','training.csv']}})
    state_path.unlink(missing_ok=True)
    return {int(k):v for k,v in arrays.items()}
