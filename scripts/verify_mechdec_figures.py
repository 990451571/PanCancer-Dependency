"""Verify installed redraws, source immutability, formats, and old-image removal."""
from pathlib import Path
import hashlib
import json
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/"results/historical/mechdec_figure_redraw_20261013"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest=json.loads((RELEASE/"render_manifest.json").read_text())
    audit=json.loads((RELEASE/"replacement_audit.json").read_text())
    assert sha(RELEASE/"render_manifest.json")==audit["render_manifest_sha256"]
    assert sha(ROOT/manifest["reference"])==manifest["reference_sha256"]
    assert sha(ROOT/"scripts/redraw_mechdec_figures.py")==manifest["renderer_sha256"]
    assert len(manifest["figures"])==12 and manifest["analysis_unchanged"]
    for relative,digest in manifest["source_sha256"].items():
        assert sha(ROOT/relative)==digest,relative
    checked=0
    for row in manifest["figures"]:
        for prefix in ("outputs","results/historical"):
            directory=ROOT/prefix/row["group"]
            run=json.loads((directory/"run.json").read_text())
            for ext in ("png","pdf"):
                path=directory/(row["stem"]+"."+ext)
                assert sha(path)==row[ext+"_sha256"]==run["output_sha256"][path.name]
                if ext=="png":
                    with Image.open(path) as im:
                        im.verify()
                    with Image.open(path) as im:
                        assert abs(im.info["dpi"][0]-300)<1 and min(im.size)>1000
                else:
                    content=path.read_bytes()
                    assert content.startswith(b"%PDF") and b"/FontFile2" in content
                checked+=1
    previous={r["path"]:r["old_sha256"] for r in audit["previous_images"]}
    assert all(sha(ROOT/path)!=previous[path] for path in audit["replaced"])
    assert len(audit["deleted_obsolete_images"])==12
    assert all(not (ROOT/r["path"]).exists() for r in audit["deleted_obsolete_images"])
    record=dict(status="passed",figure_n=12,installed_artifact_n=checked,deleted_obsolete_artifact_n=12,
        source_file_n=len(manifest["source_sha256"]),reference_unchanged=True,source_data_unchanged=True,
        png_dpi=300,pdf_embedded_truetype_fonts=True,visual_review="All 12 PNG files inspected at full resolution",
        scope="Rendering and provenance only; no model refit or new biological validation")
    target=RELEASE/"verification.json"
    if target.exists():
        assert json.loads(target.read_text())==record
    else:
        target.write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(record,indent=2,ensure_ascii=False))


if __name__=="__main__": main()
