"""Check public inventory, source grouping, feature consistency and excluded material."""
from pathlib import Path
import csv
import hashlib
import json
import re
import sys

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from data_protocol import Bank


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path=ROOT/"release_manifest.json"
    manifest=json.loads(manifest_path.read_text())
    checks=0
    for name,entry in manifest["files"].items():
        path=ROOT/name
        assert path.is_file(),name
        assert digest(path)==entry["sha256"],name
        checks+=1
    omitted={".tex",".bib",".bst",".cls",".pdf",".docx",".pptx",".zip",".gz",".pem",".key"}
    for name in manifest["files"]:
        path=ROOT/name
        assert path.suffix.lower() not in omitted,path.relative_to(ROOT)
        assert "raw" not in path.relative_to(ROOT).parts,path.relative_to(ROOT)
    bank=Bank(ROOT)
    catalog=pd.read_csv(ROOT/"data/processed/multi_input_state_catalog.csv")
    assert catalog.source_uid.str.fullmatch(r"source-\d{4}").all()
    assert len(catalog)==1264 and catalog.is_change.sum()==51
    design=json.loads((ROOT/"results/frozen_design.json").read_text())
    expected=[(606,630,24),(364,382,18),(243,252,9)]
    for role,counts in zip(["train","validation","test"],expected):
        data=bank.dataset(design["roles"][role],normal_only=True)
        assert (len(design["roles"][role]),len(data.y),int(data.update.sum()))==counts
        checks+=1
    for name in ["st","et","disc","previous_st","previous_et","event_time"]:
        assert pd.to_datetime(catalog[name]).dt.year.eq(2025).all(),name
    assert len(list((ROOT/"results/cache/models").glob("*.pkl")))==980
    assert len(list((ROOT/"results/test_scores").glob("*.npz")))==980
    for path in (ROOT/"notebooks").glob("*.ipynb"):
        nb=json.loads(path.read_text(encoding="utf-8"))
        assert nb["nbformat"]==4
    print(json.dumps({"status":"PASS","inventory_files":len(manifest["files"]),
                      "checks":checks,"normal_states":len(catalog),"benign_user_updates":51,
                      "original_raw_JSON_included":False,"manuscript_included":False,
                      "models":980,"score_arrays":980},indent=2))


if __name__=="__main__":
    main()
