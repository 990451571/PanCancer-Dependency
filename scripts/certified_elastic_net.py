"""CUDA working-set Elastic Net; certificates always cover every feature."""
import math
import numpy as np
import torch


def diagnostics(x, y, w, alpha):
    penalty = alpha * .5
    residual = x @ w - y
    gradient = x.T @ residual / len(x) + penalty*w
    subgradient = torch.where(w != 0, gradient + penalty*w.sign(),
                             gradient.sign()*torch.clamp(gradient.abs()-penalty,min=0.))
    objective = residual.square().sum(0)/(2*len(x)) + penalty*w.abs().sum(0) + .5*penalty*w.square().sum(0)
    bound = subgradient.square().sum(0)/(2*penalty)
    return objective,subgradient.abs().amax(0),bound/torch.clamp(objective,min=1e-12),subgradient


def passed(cert, protocol):
    return (cert[1] <= protocol['kkt_absolute_tolerance']) & (cert[2] <= protocol['relative_suboptimality_bound_tolerance'])


def restricted_fit(x,y,w,alpha,protocol):
    gram=x.T@x/len(x); cross=x.T@y/len(x)
    lipschitz=float(gram.abs().sum(1).max())+alpha*.5
    current=w.clone(); z=w.clone(); momentum=1.
    for iteration in range(protocol['maximum_refinement_iterations']+1):
        if iteration%25==0:
            if passed(diagnostics(x,y,current,alpha),protocol).all():
                return current,iteration
        if iteration==protocol['maximum_refinement_iterations']:break
        gradient=gram@z-cross+alpha*.5*z
        step=z-gradient/lipschitz
        updated=step.sign()*torch.clamp(step.abs()-alpha*.5/lipschitz,min=0.)
        next_momentum=(1.+math.sqrt(1.+4.*momentum**2))/2.
        if torch.sum((z-updated)*(updated-current))>0:
            z,next_momentum=updated.clone(),1.
        else:
            z=updated+(momentum-1.)/next_momentum*(updated-current)
        current,momentum=updated,next_momentum
    raise RuntimeError('工作集求解未达到数值证书')


def certified_path(x, y, hx, alphas, protocol):
    """Independent targets, observed-row intercepts, historical float32 centering."""
    if x.dtype!=torch.float32 or hx.dtype!=torch.float32:
        raise ValueError('输入应为历史标准化float32设计')
    observed=np.isfinite(y)
    _,groups=np.unique(np.packbits(observed.T,axis=1),axis=0,return_inverse=True)
    predictions={a:np.empty((len(hx),y.shape[1]),dtype=float) for a in alphas}
    records=[]
    for group in np.unique(groups):
        columns=np.flatnonzero(groups==group);rows=np.flatnonzero(observed[:,columns[0]])
        if len(rows)<2:raise ValueError('靶点有效训练样本不足')
        gx=x[torch.as_tensor(rows,device='cuda')]
        feature_mean=gx.mean(0)
        gx=(gx-feature_mean).double();ghx=(hx-feature_mean).double()
        for start in range(0,len(columns),protocol['target_batch_size']):
            target=columns[start:start+protocol['target_batch_size']]
            raw=y[np.ix_(rows,target)];mean=raw.mean(0)
            gy=torch.as_tensor(raw-mean,dtype=torch.float32,device='cuda').double()
            weight=torch.zeros((x.shape[1],len(target)),dtype=torch.float64,device='cuda')
            for alpha in sorted(alphas,reverse=True):
                iterations=0;max_active=0
                for working_pass in range(protocol['maximum_working_set_passes']+1):
                    cert=diagnostics(gx,gy,weight,alpha)
                    if passed(cert,protocol).all():break
                    if working_pass==protocol['maximum_working_set_passes']:
                        raise RuntimeError(f'全特征证书未通过 alpha={alpha} KKT={cert[1].max().item()}')
                    violation=cert[3].abs()
                    _,indices=torch.topk(violation,min(protocol['working_set_add_per_target'],len(weight)),dim=0)
                    active=(weight!=0).any(1)
                    active[indices.flatten()]=True
                    index=torch.where(active)[0]
                    max_active=max(max_active,len(index))
                    fitted,count=restricted_fit(gx[:,index],gy,weight[index],alpha,protocol)
                    weight[index]=fitted;iterations+=count
                predictions[alpha][:,target]=(ghx@weight).cpu().numpy()+mean
                objective,kkt,relative=[v.cpu().numpy() for v in cert[:3]]
                for j,column in enumerate(target):
                    records.append({'target_index':int(column),'alpha':alpha,'objective':float(objective[j]),
                        'kkt':float(kkt[j]),'relative_bound':float(relative[j]),'iterations':iterations,
                        'working_set_passes':working_pass,'max_active_feature_n':max_active,
                        'observed_train_n':len(rows)})
    return predictions,records
