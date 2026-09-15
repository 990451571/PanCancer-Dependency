"""Freeze and extract the prespecified renal validation slice from Project Score v1.

This script reads two archived Sanger matrices but deliberately does not calculate
prediction performance or use outcomes to select models, genes, or hyperparameters.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
import struct
import urllib.request
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ARCHIVE_URL = "https://cmp.cog.sanger.ac.uk/download/Project_score_archive_data.zip"
MEMBERS = {
    "scaled_bayesian_factor": "Project_score_archive_data/Release1/EssentialityMatrices/03_scaledBayesianFactors.tsv",
    "binary_dependency": "Project_score_archive_data/Release1/EssentialityMatrices/04_binaryDepScores.tsv",
}


class HttpRangeReader(io.RawIOBase):
    """Seekable HTTP reader used by ZipFile without downloading the 615 MB archive."""

    def __init__(self, url):
        self.url = url
        self.position = 0
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise RuntimeError("归档服务器不支持 HTTP Range，停止以避免整包下载")
            self.size = int(response.headers["Content-Range"].split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"未知 whence：{whence}")
        if position < 0:
            raise ValueError("不能定位到负偏移")
        self.position = position
        return position

    def read(self, size=-1):
        remaining = self.size - self.position
        size = remaining if size is None or size < 0 else min(size, remaining)
        if size <= 0:
            return b""
        data = self.read_at(self.position, size)
        self.position += len(data)
        return data

    def read_at(self, position, size):
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={position}-{position + size - 1}"})
        with urllib.request.urlopen(request, timeout=180) as response:
            if response.status != 206:
                raise RuntimeError("归档 Range 请求未返回 206")
            data = response.read()
        if len(data) != size:
            raise IOError(f"Range 响应长度错误：预期 {size}，得到 {len(data)}")
        return data


def read_member_once(reader, info):
    """Fetch a member's compressed span in one request and validate its CRC."""
    local_header = reader.read_at(info.header_offset, 30)
    fields = struct.unpack("<4s5H3L2H", local_header)
    if fields[0] != b"PK\x03\x04":
        raise zipfile.BadZipFile("成员本地文件头签名错误")
    name_length, extra_length = fields[-2:]
    start = info.header_offset + 30 + name_length + extra_length
    compressed = reader.read_at(start, info.compress_size)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -zlib.MAX_WBITS)
    elif info.compress_type == zipfile.ZIP_STORED:
        data = compressed
    else:
        raise NotImplementedError(f"不支持 ZIP 压缩方法 {info.compress_type}")
    if len(data) != info.file_size or (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC:
        raise zipfile.BadZipFile("成员长度或 CRC 校验失败")
    return data


def extract_columns(data, member, model_names, destination):
    digest = hashlib.sha256()
    seen_genes = set()
    duplicate_genes = set()
    values = {}
    with io.BytesIO(data) as source, gzip.open(destination, "wt", newline="") as output:
        header_bytes = source.readline()
        digest.update(header_bytes)
        header = header_bytes.decode("utf-8-sig").rstrip("\r\n").split("\t")
        missing = sorted(set(model_names) - set(header))
        if missing:
            raise ValueError(f"{member} 缺少预先指定模型：{missing}")
        indices = [0] + [header.index(name) for name in model_names]
        writer = csv.writer(output)
        writer.writerow([header[index] for index in indices])
        for line in source:
            digest.update(line)
            fields = line.decode("utf-8").rstrip("\r\n").split("\t")
            gene = fields[0]
            if gene in seen_genes:
                duplicate_genes.add(gene)
            seen_genes.add(gene)
            selected = [fields[index] for index in indices]
            writer.writerow(selected)
            values[gene] = selected[1:]
    return {
        "member": member,
        "uncompressed_sha256": digest.hexdigest(),
        "gene_n": len(seen_genes),
        "duplicate_gene_n": len(duplicate_genes),
        "model_n": len(model_names),
    }, values


def main():
    parser = argparse.ArgumentParser(description="冻结 Sanger 肾癌外部验证方案并提取指定标签")
    parser.add_argument("--audit-dir", type=Path, default=Path("outputs/external_dependency_audit_v1"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("data/processed/depmap_baseline_24q4_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sanger_validation_frozen_v1"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir}")

    print("【阶段 1/3】冻结候选名单与验证规则", flush=True)
    candidates = pd.read_csv(args.audit_dir / "renal_candidates.csv")
    candidates = candidates.sort_values("model_name", kind="stable").reset_index(drop=True)
    expected = {"769-P", "LB1047-RCC", "RCC-FG2"}
    if set(candidates.model_name) != expected or len(candidates) != 3:
        raise ValueError("肾癌候选名单发生变化，拒绝静默改变验证集合")
    if candidates.strict_ccrcc.sum() != 2 or candidates.in_current_cohort.fillna(True).any():
        raise ValueError("候选亚型或训练集隔离状态不符合已审计结果")
    model_names = candidates.model_name.tolist()
    baseline_genes = set(np.load(args.baseline_dir / "matrices.npz", allow_pickle=False)["genes"].astype(str))
    coverage = pd.read_csv(args.baseline_dir / "gene_coverage.csv").set_index("Gene")
    nonessential = set(coverage.index[~coverage.DepMap_common_essential.astype(bool)])

    temporary = args.output_dir.with_name(args.output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    print("【阶段 2/3】按字节范围读取 Project Score Release 1，不下载完整归档", flush=True)
    matrix_records = {}
    extracted = {}
    try:
        reader = HttpRangeReader(ARCHIVE_URL)
        with zipfile.ZipFile(reader) as archive:
            for label, member in MEMBERS.items():
                print(f"  提取 {label}：仅基因列和 3 个预指定模型", flush=True)
                data = read_member_once(reader, archive.getinfo(member))
                record, values = extract_columns(
                    data, member, model_names, temporary / f"renal_{label}.csv.gz")
                matrix_records[label] = record
                extracted[label] = values

        sanger_genes = set(extracted["scaled_bayesian_factor"])
        if sanger_genes != set(extracted["binary_dependency"]):
            raise ValueError("连续与二元标签的基因集合不一致")
        exact_overlap = sanger_genes & baseline_genes
        evaluation_genes = exact_overlap & nonessential

        relation_counts = {"binary_1_and_scaled_gt_0": 0, "binary_1_total": 0,
                           "binary_0_and_scaled_le_0": 0, "binary_0_total": 0}
        for gene in sanger_genes:
            for score_text, binary_text in zip(extracted["scaled_bayesian_factor"][gene],
                                               extracted["binary_dependency"][gene]):
                if score_text == "" or binary_text == "":
                    continue
                score, binary = float(score_text), int(float(binary_text))
                if binary == 1:
                    relation_counts["binary_1_total"] += 1
                    relation_counts["binary_1_and_scaled_gt_0"] += int(score > 0)
                elif binary == 0:
                    relation_counts["binary_0_total"] += 1
                    relation_counts["binary_0_and_scaled_le_0"] += int(score <= 0)
                else:
                    raise ValueError(f"二元依赖标签不是 0/1：{binary_text}")
        if (relation_counts["binary_1_and_scaled_gt_0"] != relation_counts["binary_1_total"] or
                relation_counts["binary_0_and_scaled_le_0"] != relation_counts["binary_0_total"]):
            raise ValueError("历史二元标签与 scaled Bayesian factor 的方向不一致")

        candidate_columns = ["model_id", "model_name", "matched_broad_id", "strict_ccrcc",
                             "broad_expression_row_available", "broad_copy_number_row_available",
                             "RNASeq Sanger Cell Lines", "RNASeq Broad Cell Lines"]
        candidates[candidate_columns].to_csv(temporary / "frozen_models.csv", index=False)
        protocol = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "frozen_before_prediction_evaluation",
            "source": {"archive_url": ARCHIVE_URL, "archive_bytes": 614764837,
                       "release": "Project Score archive Release1", "matrices": matrix_records},
            "model_selection": {
                "rule": "Sanger CRISPR cell line, tissue exactly Kidney, absent from current 873-model cohort",
                "models": model_names, "model_n": len(model_names), "strict_ccrcc_n": 2,
                "selection_used_functional_outcomes": False},
            "gene_universe": {
                "mapping": "exact gene symbol only; no outcome-dependent remapping",
                "sanger_gene_n": len(sanger_genes), "baseline_gene_n": len(baseline_genes),
                "exact_overlap_n": len(exact_overlap),
                "primary_non_common_essential_n": len(evaluation_genes)},
            "endpoint_orientation_check": relation_counts,
            "evaluation": {
                "primary_scope": "exact-overlap genes excluding DepMap 24Q4 common essentials",
                "continuous_truth": "negative of Release1 scaled Bayesian factor; lower means more dependent",
                "binary_truth": "Release1 binary dependency score; 1 means dependent",
                "primary_metrics": ["NDCG@10", "binary dependency precision@10", "top-10 overlap"],
                "secondary_metric": "Spearman correlation across the fixed primary gene universe",
                "comparators": ["training-gene mean", "expression-only ridge alpha=100000"],
                "hyperparameter_rule": "reuse alpha=100000; no tuning on the three validation outcomes",
                "training_rule": "train on current cohort excluding all Kidney models and their patients",
                "phase_a": "769-P using existing Broad expression; direct feasibility evaluation for one model",
                "phase_b": "LB1047-RCC and RCC-FG2 only after expression-domain mapping fitted without their functional outcomes",
                "aggregation": "report every model separately; no inferential p-value or ccRCC-wide estimate at n=2"},
            "interpretation_limits": [
                "All three models also have Broad CRISPR data, so biological-sample independence is absent.",
                "The Release1 Sanger processing is independent of Broad Chronos but is an older assay pipeline.",
                "A single direct-expression model and two ccRCC models provide feasibility evidence only.",
                "The frozen outcomes must not be used for feature, alpha, threshold, or mapping selection."]}
        (temporary / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
        temporary.rename(args.output_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    print(f"【阶段 3/3】冻结完成：模型 3｜明确 ccRCC 2｜精确重叠基因 {len(exact_overlap)}｜主分析基因 {len(evaluation_genes)}", flush=True)
    print("【未执行】尚未生成预测、计算指标或调整模型。", flush=True)
    print(f"【保存位置】{args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
