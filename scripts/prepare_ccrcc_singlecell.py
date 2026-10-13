"""Download and inventory the author-annotated GSE159115 expression cohort."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
import time
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE159nnn/GSE159115"
SOURCES = {
    "GSE159115_RAW.tar": BASE + "/suppl/GSE159115_RAW.tar",
    "GSE159115_ccRCC_anno.csv.gz": BASE + "/suppl/GSE159115_ccRCC_anno.csv.gz",
    "GSE159115_normal_anno.csv.gz": BASE + "/suppl/GSE159115_normal_anno.csv.gz",
    "GSE159115_family.soft.gz": BASE + "/soft/GSE159115_family.soft.gz",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, path):
    if path.exists():
        return
    part = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "PanCancerDependency/1.0 public research"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            part.rename(path)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def soft_samples(path):
    rows, current = [], None
    with gzip.open(path, "rt") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                if current is not None:
                    rows.append(current)
                current = {"accession": line.split(" = ", 1)[1]}
            elif current is not None and line.startswith("!Sample_title = "):
                current["sample_id"] = line.split(" = ", 1)[1]
            elif current is not None and line.startswith("!Sample_characteristics_ch1 = "):
                value = line.split(" = ", 1)[1]
                if ": " in value:
                    key, value = value.split(": ", 1)
                    current[key] = value
    if current is not None:
        rows.append(current)
    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=root / "data/raw/gse159115")
    args = parser.parse_args()
    directory = args.raw_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "source_manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        for name, item in previous["files"].items():
            if sha256(directory / name) != item["sha256"]:
                raise ValueError(f"公开输入哈希发生变化: {name}")
        print("【来源校验】已下载版本全部通过", flush=True)
        return
    for name, url in SOURCES.items():
        print(f"【下载公开输入】{name}", flush=True)
        download(url, directory / name)
    matrix_dir = directory / "counts"
    matrix_dir.mkdir(exist_ok=True)
    with tarfile.open(directory / "GSE159115_RAW.tar") as archive:
        for member in archive.getmembers():
            if not member.isfile() or Path(member.name).name != member.name or not member.name.endswith(".h5"):
                raise ValueError(f"拒绝意外归档成员: {member.name}")
            destination = matrix_dir / member.name
            with archive.extractfile(member) as source, destination.open("wb") as out:
                while block := source.read(1024 * 1024):
                    out.write(block)
    samples = soft_samples(directory / "GSE159115_family.soft.gz")
    samples.to_csv(directory / "samples.csv", index=False)
    print(samples.to_string(index=False), flush=True)
    for name in ["GSE159115_ccRCC_anno.csv.gz", "GSE159115_normal_anno.csv.gz"]:
        table = pd.read_csv(directory / name, index_col=0)
        print(name, table.shape, table.columns.tolist(), flush=True)
        print(table.head(3).to_string(), flush=True)
        for column in table:
            if table[column].nunique() < 40:
                print(column, table[column].value_counts().to_dict(), flush=True)
    files = {name: {"url": url, "bytes": (directory / name).stat().st_size,
                    "sha256": sha256(directory / name)} for name, url in SOURCES.items()}
    for path in sorted(matrix_dir.glob("*.h5")):
        files[str(path.relative_to(directory))] = {"archive": "GSE159115_RAW.tar",
                                                   "bytes": path.stat().st_size, "sha256": sha256(path)}
    files["samples.csv"] = {"derived_from": "GSE159115_family.soft.gz", "sha256": sha256(directory / "samples.csv")}
    manifest.write_text(json.dumps({"accession": "GSE159115", "files": files,
                                    "retrieved_date": "2026-10-10", "sample_n": len(samples)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
