"""Apply Chinese descriptive filenames to the approved 12 individual figures."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/"results/historical/mechdec_figure_redraw_20261013"
NAMES={
 "figure1_validation_and_evidence_ceiling": "图01_模型验证表现与患者证据边界",
 "figure2_candidate_evidence_matrix": "图02_ccRCC候选基因多层证据矩阵",
 "figure3_dependency_vs_normal_kidney_exposure": "图03_肿瘤基因依赖与正常肾表达风险",
 "figure4_baseline_generalization": "图04_基线模型跨癌系与跨平台表现比较",
 "figure1_pcr_transfer_stability": "图05_PCR患者迁移的排序稳定性与映射敏感性",
 "figure2_candidate_cell_sources": "图06_GSE159115候选基因的细胞来源与表达检出",
 "figure3_mixture_and_source_audit": "图07_细胞混合对排序的影响与样本标签核查",
 "paired_controls": "图08_PCR排序不稳定性的来源对照",
 "functional_stability": "图09_排序稳定性与功能预测表现的关系",
 "boundary_stability": "图10_候选排名边界间隔与排序稳定性",
 "candidate_cell_sources": "图11_GSE178481独立队列候选基因的细胞来源",
 "paired_expression": "图12_肿瘤与正常肾的配对供者表达比较",
}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text(encoding="utf-8"))
def dump(path,record): path.write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


def main():
    mapping_path=RELEASE/"chinese_filename_mapping.json"
    if mapping_path.exists():
        print("中文命名已完成；未重复修改文件。");return
    manifest=load(RELEASE/"render_manifest.json")
    audit=load(RELEASE/"replacement_audit.json")
    mapping=[];plan=[]
    roots=[ROOT/"outputs",ROOT/"results/historical",ROOT/"outputs/mechdec_redraw_20261013_review2"]
    for record in manifest["figures"]:
        old=record["stem"];new=NAMES[old]
        mapping.append(dict(group=record["group"],old_stem=old,chinese_stem=new))
        for base in roots:
            for ext in ("png","pdf"):
                source=base/record["group"]/(old+"."+ext)
                dest=source.with_name(new+"."+ext)
                assert source.resolve().is_relative_to(ROOT.resolve()) and dest.resolve().is_relative_to(ROOT.resolve())
                assert source.exists() and not dest.exists(),str(source)
                assert sha(source)==record[ext+"_sha256"]
                plan.append((source,dest))
        record["original_stem"]=old;record["stem"]=new
    # Entire rename plan validated before any mutation.
    translated={str(s.relative_to(ROOT)):str(d.relative_to(ROOT)) for s,d in plan}
    for source,dest in plan: source.rename(dest)
    for group in sorted({r["group"] for r in mapping}):
        out=ROOT/"outputs"/group;snapshot=ROOT/"results/historical"/group
        lookup={r["old_stem"]:r["chinese_stem"] for r in mapping if r["group"]==group}
        name_lookup={old+"."+ext:new+"."+ext for old,new in lookup.items() for ext in ("png","pdf")}
        caption_path=out/"figure_manifest.csv"
        if caption_path.exists():
            with caption_path.open(encoding="utf-8",newline="") as handle:
                reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
            for row in rows: row["stem"]=lookup.get(row["stem"],row["stem"])
            with caption_path.open("w",encoding="utf-8",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            shutil.copyfile(caption_path,snapshot/caption_path.name)
        run=load(out/"run.json")
        run["filename_revision"]={"language":"Chinese","date":"2026-10-13","image_contents_unchanged":True,"mapping":lookup}
        run["output_sha256"]={name_lookup.get(name,name):sha(out/name_lookup.get(name,name)) for name in run["output_sha256"]}
        dump(out/"run.json",run);shutil.copyfile(out/"run.json",snapshot/"run.json")
        if (snapshot/"snapshot.json").exists():
            snap=load(snapshot/"snapshot.json")
            snap["upstream_run_sha256"]=sha(out/"run.json")
            snap["included_sha256"]={name_lookup.get(name,name):sha(snapshot/name_lookup.get(name,name)) for name in snap["included_sha256"]}
            snap["filename_revision"]=run["filename_revision"]
            dump(snapshot/"snapshot.json",snap)
    removed=[]
    for path in [RELEASE/"figure_overview.jpg",ROOT/"outputs/mechdec_redraw_20261013_review2/figure_overview.jpg"]:
        assert path.resolve().is_relative_to(ROOT.resolve())
        if path.exists(): removed.append(str(path.relative_to(ROOT)));path.unlink()
    manifest["filename_revision"]={"language":"Chinese","overview_removed":True,"individual_figure_n":12}
    dump(RELEASE/"render_manifest.json",manifest)
    dump(ROOT/"outputs/mechdec_redraw_20261013_review2/render_manifest.json",manifest)
    audit["render_manifest_sha256"]=sha(RELEASE/"render_manifest.json")
    audit["replaced"]=[translated.get(name,name) for name in audit["replaced"]]
    for row in audit["previous_images"]:
        row["original_path"]=row["path"];row["path"]=translated.get(row["path"],row["path"])
    audit["filename_revision"]={"image_contents_unchanged":True,"deleted_overviews":removed}
    dump(RELEASE/"replacement_audit.json",audit)
    dump(mapping_path,{"status":"complete","figures":mapping,"renamed_file_n":len(plan),"deleted_overviews":removed})
    readme=ROOT/"README.md";text=readme.read_text(encoding="utf-8")
    for row in mapping:
        for ext in ("png","pdf"):
            text=text.replace(row["group"]+"/"+row["old_stem"]+"."+ext,row["group"]+"/"+row["chinese_stem"]+"."+ext)
    text=text.replace("[12图总览](results/historical/mechdec_figure_redraw_20261013/figure_overview.jpg)、","")
    text=text.replace("全部新图仍保存在原路径：","全部新图仍保存在原目录，已按内容改为中文文件名（总览图已删除）：")
    readme.write_text(text,encoding="utf-8")
    print(f"完成：12张独立结果图采用中文名称，PNG/PDF及清单同步；总览图删除{len(removed)}份。")


if __name__=="__main__": main()
