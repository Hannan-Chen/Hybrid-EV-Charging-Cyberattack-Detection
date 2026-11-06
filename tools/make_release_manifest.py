"""Hash public source artifacts, excluding private/raw inputs and local build products."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    files={}
    excluded={".tex",".bib",".bst",".cls",".pdf",".docx",".pptx",".zip",".gz",".pem",".key",".log",".pyc"}
    for path in sorted(ROOT.rglob("*")):
        rel=path.relative_to(ROOT)
        if not path.is_file() or set(rel.parts)&{".git","__pycache__",".ipynb_checkpoints",".venv"}:
            continue
        if rel.as_posix()=="release_manifest.json" or path.suffix.lower() in excluded or "raw" in rel.parts:
            continue
        files[rel.as_posix()]={"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size}
    provenance=json.loads((ROOT/"data/public_release_provenance.json").read_text())
    result={"version":"1.0.0","release_date":provenance["release_date"],
            "scope":"Public derived benchmark, code and standalone results; no manuscript or raw ACN export.",
            "manifest_self_excluded":True,"files":files}
    (ROOT/"release_manifest.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("Manifest:",len(files),"files;",sum(v["bytes"] for v in files.values()),"bytes")


if __name__=="__main__":
    main()
