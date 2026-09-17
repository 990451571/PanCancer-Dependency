#!/usr/bin/env python3
"""GPU pretrain the frozen Exp-DeepDEP expression autoencoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from advanced_model_methods import pretrain_expression_autoencoder


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=root / "data/processed/tcga_pancan_deepdep_input_v1")
    parser.add_argument("--protocol", type=Path,
                        default=root / "configs/advanced_model_benchmark_protocol_20260917.json")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "data/processed/tcga_pancan_deepdep_pretrain_v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；自编码器不会退回CPU")
    print(f"【计算设备】{torch.cuda.get_device_name(0)}｜CUDA｜float32", flush=True)
    print("【阶段 1/3】校验无标签泛癌输入与冻结协议", flush=True)
    audit = json.loads((args.input_dir / "audit.json").read_text())
    matrix_path = args.input_dir / "tcga_pancan_expression.npz"
    if sha256(matrix_path) != audit["output_sha256"][matrix_path.name]:
        raise ValueError("TCGA预训练输入哈希错误")
    if audit["dependency_labels_read"] or audit["tcga_kirc_expression_values_read"]:
        raise ValueError("预训练输入隔离状态异常")
    protocol = json.loads(args.protocol.read_text())
    expected = protocol["methods"][3]
    if args.epochs != expected["maximum_epochs"]:
        raise ValueError("自编码器epoch必须与冻结DeepDEP上限一致")
    with np.load(matrix_path, allow_pickle=False) as archive:
        expression = archive["expression"]
        sample_n, feature_n = expression.shape
    print(f"【预训练队列】非KIRC肿瘤 {sample_n}｜表达特征 {feature_n}", flush=True)
    if args.dry_run:
        print("【检查通过】未训练、未写入结果。", flush=True)
        return
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir}")

    print("【阶段 2/3】GPU训练官方结构表达自编码器", flush=True)
    encoder, losses = pretrain_expression_autoencoder(
        expression, args.seed, args.epochs, args.batch_size, report_every=10)
    print("【阶段 3/3】保存编码器和训练审计", flush=True)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent))
    try:
        torch.save(encoder.state_dict(), temporary / "expression_encoder.pt")
        pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "reconstruction_mse": losses}).to_csv(
            temporary / "training_loss.csv", index=False)
        run = {
            "status": "exp_deepdep_expression_autoencoder_pretrained",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "dtype": "float32",
            "sample_n": sample_n,
            "feature_n": feature_n,
            "architecture": [feature_n, 500, 200, 50, 200, 500, feature_n],
            "activation": "ReLU",
            "optimizer": "Adam(lr=0.001)",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "initial_mse": losses[0],
            "final_mse": losses[-1],
            "dependency_labels_read": False,
            "tcga_kirc_included": False,
            "input_sha256": {
                "matrix": sha256(matrix_path),
                "audit": sha256(args.input_dir / "audit.json"),
                "protocol": sha256(args.protocol),
                "script": sha256(Path(__file__)),
                "methods_script": sha256(Path(__file__).with_name("advanced_model_methods.py")),
            },
        }
        run["output_sha256"] = {path.name: sha256(path) for path in temporary.iterdir()}
        (temporary / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
        temporary.rename(args.output_dir)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(f"【完成】耗时 {(time.monotonic() - started) / 60:.1f}分钟｜最终重建MSE {losses[-1]:.6f}", flush=True)
    print(f"【结果】{args.output_dir}", flush=True)
    print("【结论边界】自编码器只学习无标签表达表示；重建误差不是依赖预测准确率。", flush=True)


if __name__ == "__main__":
    main()
