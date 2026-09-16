#!/usr/bin/env python3
"""Verify the external raw and frozen TCGA inputs required for a full rerun."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from runtime_paths import source_project_root


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    repository = Path(__file__).resolve().parents[1]
    external = source_project_root(repository)
    manifest = json.loads((repository / "configs/public_data_sources.json").read_text())
    print(f"【外部输入根目录】{external}", flush=True)
    checked_bytes = 0
    checked_files = 0

    print("【阶段 1/2】核验 DepMap 24Q4 原始文件", flush=True)
    raw = external / "data/raw/depmap_24q4"
    for name, expected in manifest["depmap_24q4"]["files"].items():
        path = raw / name
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size != expected["bytes"] or digest(path, "md5") != expected["md5"]:
            raise ValueError(f"DepMap文件版本或内容不一致：{path}")
        checked_files += 1
        checked_bytes += path.stat().st_size
        print(f"  【通过】{name}", flush=True)

    print("【阶段 2/2】核验冻结 TCGA 表达、患者分组与HGNC映射", flush=True)
    for relative, expected in manifest["tcga_kirc"]["required_external_files"].items():
        path = external / relative
        if not path.exists():
            raise FileNotFoundError(path)
        if digest(path, "sha256") != expected["sha256"]:
            raise ValueError(f"TCGA/HGNC输入版本或内容不一致：{path}")
        checked_files += 1
        checked_bytes += path.stat().st_size
        print(f"  【通过】{relative}", flush=True)
    print(f"【输入就绪】文件 {checked_files}｜核验 {checked_bytes / 1024**3:.2f} GiB", flush=True)
    print("【结论边界】这里只核验主流程外部输入；Sanger、DRIVE、GTEx和HPA由各阶段运行记录继续固定。", flush=True)


if __name__ == "__main__":
    main()
