"""Resumable development; final test is opened only after selection is hashed."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import pickle
import platform
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.externals import cloudpickle

from data_protocol import Bank, TYPES, digest, save_json
from models import Fitted, fit_u, robust_scale
from protocol import (FAMILIES, SEEDS, TASKS, members, evaluate, exact_threshold,
                      freeze, load, registry, signature)

_BANKS, _NORMAL = {}, {}


def bank_for(root):
    key = str(Path(root).resolve())
    if key not in _BANKS:
        _BANKS[key] = Bank(key)
    return _BANKS[key]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f".{os.getpid()}.tmp")
    joblib.dump(value, temp, compress=3)
    temp.replace(path)


def model_dump(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f".{os.getpid()}.tmp")
    with temp.open("wb") as f:
        cloudpickle.dump(model, f)
    temp.replace(path)


def original_u(root, seed):
    key = (str(root), seed)
    if key not in _NORMAL:
        with (Path(root)/f"results/cache/models/u_{seed}.pkl").open("rb") as f:
            _NORMAL[key] = cloudpickle.load(f)
    return _NORMAL[key]


def baseline_seed(root, frozen, seed):
    root = Path(root)
    destination = root/f"results/cache/u_{seed}.joblib"
    if destination.exists():
        return joblib.load(destination)["summary"]
    bank = bank_for(root)
    normal = bank.dataset(frozen["roles"]["train"], normal_only=True)
    vn = bank.dataset(frozen["roles"]["validation"], normal_only=True)
    model = fit_u(normal, seed)
    train_scores, val_normal_scores = model.score(normal.x), model.score(vn.x)
    rows, values = {}, {}
    threshold = exact_threshold(vn, val_normal_scores)
    for kind in TASKS:
        cond = next(c for c in frozen["conditions"] if c["train_type"] == kind)
        keys = cond["repeats"][0]["validation_keys"]
        data = bank.dataset(frozen["roles"]["validation"], set(keys), members(kind))
        scores = model.score(data.x)
        rows[kind] = evaluate(data, scores, threshold, kind)
        values[kind] = {"scores": scores, "keys": data.key}
    path = root/f"results/cache/models/u_{seed}.pkl"
    model_dump(path, model)
    payload = {"summary": {"seed": seed, "seconds": model.seconds, "threshold": threshold},
               "train_scores": train_scores, "normal_validation_scores": val_normal_scores,
               "validation": values, "metrics": rows, "normalization": robust_scale(train_scores),
               "model_sha256": digest(path), "signature": frozen["signature"]}
    dump(destination, payload)
    return payload["summary"]


def baseline(root, workers):
    frozen = load(root)
    rows = Parallel(n_jobs=min(workers, 5), backend="loky")(delayed(baseline_seed)(root, frozen, s) for s in SEEDS)
    pd.DataFrame(rows).to_csv(Path(root)/"results/normal_only_baseline.csv", index=False)
    print("NORMAL_ONLY_READY", len(rows), flush=True)


def repeat_for(condition, seed):
    return next(r for r in condition["repeats"] if r["seed"] == seed)


def fit_id(condition, config, seed, ablation=None):
    return condition["id"]+"__"+config["id"]+"__s"+str(seed)+(f"__{ablation}" if ablation else "")


def task(root, frozen, condition, config, seed, ablation=None):
    root = Path(root)
    key = fit_id(condition, config, seed, ablation)
    dest = root/"results/cache/development"/(key+".joblib")
    if dest.exists():
        cached = joblib.load(dest)
        assert cached["signature"] == frozen["signature"]
        return cached["summary"]
    start = time.perf_counter()
    rep, kind = repeat_for(condition, seed), condition["train_type"]
    try:
        bank = bank_for(root)
        train = bank.dataset(frozen["roles"]["train"], set(rep["train_keys"]), members(kind))
        val = bank.dataset(frozen["roles"]["validation"], set(rep["validation_keys"]), members(kind))
        assert set(train.source).isdisjoint(val.source)
        assert set(train.key[train.y == 1]) == set(rep["train_keys"])
        assert len(set(train.source[train.y == 1])) == len(rep["train_keys"])
        model = Fitted(config, seed).fit(train, root, ablation)
        scores = model.score(val.x)
        vn = val.take(val.y == 0)
        threshold = exact_threshold(vn, scores[val.y == 0])
        row = evaluate(val, scores, threshold, kind)
        train_normal_scores = model.score(train.x[train.y == 0])
        path = root/"results/cache/models"/(key+".pkl")
        model_dump(path, model)
        summary = {**row, "fit_id": key, "condition_id": condition["id"], "config_id": config["id"],
                   "method": config["family"], "seed": seed, "status": "ok", "ablation": ablation,
                   "training_seconds": model.seconds, "pretrain_seconds": model.pretrain_seconds,
                   "parameter_count": model.parameter_count, "total_seconds": time.perf_counter()-start}
        payload = {"signature": frozen["signature"], "summary": summary, "config": config,
                   "scores": scores, "keys": val.key, "labels": val.y, "updates": val.update,
                   "sources": val.source, "threshold": threshold,
                   "normalization": robust_scale(train_normal_scores), "training_normal_scores": train_normal_scores,
                   "fit_attack_keys": train.key[train.y == 1], "validation_attack_keys": val.key[val.y == 1],
                   "model_sha256": digest(path), "ablation": ablation}
    except Exception as e:
        summary = {"fit_id": key, "condition_id": condition["id"], "config_id": config["id"],
                   "method": config["family"], "seed": seed, "status": "fit_failed", "ablation": ablation,
                   "change_class_macro_f1": 0., "update_fpr": 1., "normal_fpr": 1.}
        payload = {"signature": frozen["signature"], "summary": summary, "error": str(e),
                   "traceback": traceback.format_exc()}
        print("FIT_FAILED", key, repr(e), flush=True)
    dump(dest, payload)
    return summary


def parallel_row(root, frozen, condition, config, seed, weight):
    key = fit_id(condition, config, seed)
    base = joblib.load(Path(root)/"results/cache/development"/(key+".joblib"))
    method = "U+"+config["family"]
    cid = config["id"]+"_w"+str(weight)
    if base["summary"]["status"] != "ok":
        return {**base["summary"], "method": method, "config_id": cid, "fusion_weight": weight}
    u = joblib.load(Path(root)/f"results/cache/u_{seed}.joblib")
    uv = u["validation"][condition["train_type"]]
    assert np.array_equal(uv["keys"], base["keys"])
    mu, su = u["normalization"]; ms, ss = base["normalization"]
    scores = np.maximum((uv["scores"]-mu)/su, weight*(base["scores"]-ms)/ss)
    rep = repeat_for(condition, seed)
    data = bank_for(root).dataset(frozen["roles"]["validation"], set(rep["validation_keys"]), members(condition["train_type"]))
    threshold = exact_threshold(data.take(data.y == 0), scores[data.y == 0])
    metrics = evaluate(data, scores, threshold, condition["train_type"])
    return {**base["summary"], **metrics, "method": method, "config_id": cid, "fusion_weight": weight,
            "u_normalization": list(u["normalization"]), "s_normalization": list(base["normalization"])}


def rank(rows, configs):
    table, out = pd.DataFrame(rows), []
    for cid, group in table.groupby("config_id", sort=True):
        assert set(group.seed) == set(SEEDS) and len(group) == len(SEEDS)
        values = group.change_class_macro_f1.to_numpy(float)
        base_id = cid.split("_w")[0]
        config = next((c for c in configs if c["id"] == base_id), {"complexity": 0})
        out.append({"config_id": cid, "method": group.method.iloc[0],
                    "successful": bool((group.status == "ok").all()),
                    "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                    "minimum": float(values.min()), "robust_score": float(values.mean()-.5*values.std(ddof=1)),
                    "update_fpr": float(group.update_fpr.mean()), "normal_fpr": float(group.normal_fpr.mean()),
                    "complexity": config["complexity"]})
    return pd.DataFrame(out).sort_values(["successful","robust_score","update_fpr","normal_fpr","complexity","config_id"],
                                         ascending=[False,False,True,True,True,True]).reset_index(drop=True)


def select_condition(root, frozen, condition):
    root = Path(root)
    folder = root/"results/development"/condition["id"]
    if (folder/"selection.json").exists():
        return
    configs = frozen["registry"]
    rows = [joblib.load(root/"results/cache/development"/(fit_id(condition,c,s)+".joblib"))["summary"]
            for c in configs for s in SEEDS]
    for c in configs:
        if c["family"] in ["RF", "SDTN"]:
            for s in SEEDS:
                for w in frozen["parallel_weights"]:
                    rows.append(parallel_row(root, frozen, condition, c, s, w))
    for s in SEEDS:
        u = joblib.load(root/f"results/cache/u_{s}.joblib")
        rows.append({**u["metrics"][condition["train_type"]], "condition_id": condition["id"],
                     "config_id": "U_fixed", "method": "U", "seed": s, "status": "ok",
                     "fit_id": f"u_{s}", "training_seconds": u["summary"]["seconds"]})
    table = pd.DataFrame(rows)
    winners, rankings = [], []
    for method, group in table.groupby("method", sort=True):
        ranking = rank(group.to_dict("records"), configs)
        rankings.append(ranking)
        winners.append(ranking.iloc[0].to_dict())
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder/"candidate_seed_results.csv", index=False)
    pd.concat(rankings).to_csv(folder/"candidate_rankings.csv", index=False)
    save_json(folder/"selection.json", {"signature": frozen["signature"], "condition_id": condition["id"],
                                      "winners": winners, "test_accessed": False})
    print("SELECTED", condition["id"], flush=True)


def search(root, workers):
    f = load(root)
    # Whole-condition scheduling makes paired configuration comparisons auditable.
    for condition in f["conditions"]:
        if (Path(root)/"results/development"/condition["id"]/"selection.json").exists():
            continue
        print("SEARCH", condition["id"], "fits", len(f["registry"])*len(SEEDS), flush=True)
        Parallel(n_jobs=min(workers,16), backend="loky", batch_size=1)(
            delayed(task)(root,f,condition,c,s) for c in f["registry"] for s in SEEDS)
        select_condition(root,f,condition)


def freeze_selection(root, workers):
    root = Path(root)
    f = load(root)
    cases, ablation_jobs = [], []
    for condition in f["conditions"]:
        folder = root/"results/development"/condition["id"]
        selection = json.loads((folder/"selection.json").read_text())
        table = pd.read_csv(folder/"candidate_seed_results.csv", float_precision="round_trip")
        for winner in selection["winners"]:
            group = table[(table.method == winner["method"]) & (table.config_id == winner["config_id"])]
            for row in group.to_dict("records"):
                if row["status"] != "ok":
                    continue
                case = {"condition_id": condition["id"], "train_type": condition["train_type"],
                        "budget": condition["budget"], "method": row["method"], "seed": int(row["seed"]),
                        "config_id": row["config_id"], "fit_id": row["fit_id"],
                        "threshold": row["threshold"], "validation_score": row["change_class_macro_f1"],
                        "attack_train_count": repeat_for(condition,int(row["seed"]))["attack_train_count"],
                        "attack_label_cost": repeat_for(condition,int(row["seed"]))["attack_label_cost"],
                        "deployable_fit": bool(winner["successful"]), "ablation": None}
                if row["method"].startswith("U+"):
                    payload = joblib.load(root/"results/cache/development"/(row["fit_id"]+".joblib"))
                    u = joblib.load(root/f"results/cache/u_{case['seed']}.joblib")
                    case.update(fusion_weight=row["fusion_weight"], u_normalization=u["normalization"],
                                s_normalization=payload["normalization"], u_model_sha256=u["model_sha256"])
                model_path = root/"results/cache/models"/(case["fit_id"]+".pkl")
                case["model_sha256"] = digest(model_path)
                cases.append(case)
                if case["method"] == "SS" and condition["budget"] in f["ablation_budgets"]:
                    config = next(c for c in f["registry"] if c["id"] == case["config_id"])
                    for ablation in f["ablations"]:
                        ablation_jobs.append((condition, config, case["seed"], ablation))
    print("MATCHED_ABLATIONS", len(ablation_jobs), flush=True)
    rows = Parallel(n_jobs=min(workers,16), backend="loky", batch_size=1)(
        delayed(task)(root,f,condition,c,s,a) for condition,c,s,a in ablation_jobs)
    pd.DataFrame(rows).to_csv(root/"results/ablation_validation.csv", index=False)
    for (condition, config, seed, ablation), row in zip(ablation_jobs, rows):
        if row["status"] != "ok":
            continue
        path = root/"results/cache/models"/(row["fit_id"]+".pkl")
        rep = repeat_for(condition,seed)
        cases.append({"condition_id": condition["id"], "train_type": condition["train_type"],
                      "budget": condition["budget"], "method": "SS-"+ablation, "seed": seed,
                      "config_id": config["id"], "fit_id": row["fit_id"], "threshold": row["threshold"],
                      "validation_score": row["change_class_macro_f1"], "ablation": ablation,
                      "attack_train_count": rep["attack_train_count"], "attack_label_cost": rep["attack_label_cost"],
                      "deployable_fit": True, "model_sha256": digest(path)})
    doc = {"protocol_signature": f["signature"], "cases": cases, "test_opened": False,
           "selection_files": {p.relative_to(root).as_posix(): digest(p) for p in (root/"results/development").glob("*/selection.json")}}
    doc["signature"] = signature(doc)
    destination = root/"results/final_selection_freeze.json"
    if destination.exists():
        assert json.loads(destination.read_text())["signature"] == doc["signature"]
    else:
        save_json(destination, doc)
    print("FINAL_SELECTION_FROZEN", len(cases), doc["signature"], flush=True)


def score_model(root, frozen, fit_key):
    root = Path(root)
    target = root/"results/test_scores"/(fit_key+".npz")
    if target.exists():
        return
    data = bank_for(root).dataset(frozen["roles"]["test"])
    path = root/"results/cache/models"/(fit_key+".pkl")
    with path.open("rb") as stream:
        model = cloudpickle.load(stream)
    started = time.perf_counter()
    scores = model.score(data.x)
    batch_seconds = time.perf_counter()-started
    latency = []
    for i in range(10):
        started = time.perf_counter(); model.score(data.x[i:i+1]); latency.append(time.perf_counter()-started)
    assert np.isfinite(scores).all()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, scores=scores, keys=data.key.astype(str), labels=data.y, sources=data.source,
                        settings=data.setting.astype(str), updates=data.update, state=data.state,
                        training_seconds=float(model.seconds), batch_seconds=batch_seconds,
                        event_latency_ms=1000*float(np.median(latency)),
                        parameter_count=int(getattr(model,"parameter_count",-1)), model_sha256=digest(path))


def evaluate_test(root, workers):
    root = Path(root);f=load(root)
    selected=json.loads((root/"results/final_selection_freeze.json").read_text())
    assert signature({k:v for k,v in selected.items() if k!="signature"}) == selected["signature"]
    for name, expected in selected["selection_files"].items():
        assert digest(root/name)==expected
    cases=selected["cases"]
    unique=sorted(set(c["fit_id"] for c in cases))
    for case in cases:
        assert digest(root/"results/cache/models"/(case["fit_id"]+".pkl"))==case["model_sha256"]
    print("FROZEN_TEST_SCORING",len(unique),"models",len(cases),"cases",flush=True)
    Parallel(n_jobs=min(workers,16),backend="loky",batch_size=1)(delayed(score_model)(root,f,key) for key in unique)
    data=bank_for(root).dataset(f["roles"]["test"])
    rows=[];variant_rows=[];metadata=[]
    for index,case in enumerate(cases):
        payload=np.load(root/"results/test_scores"/(case["fit_id"]+".npz"))
        scores=payload["scores"]
        event_ms=float(payload["event_latency_ms"])
        training_seconds=float(payload["training_seconds"])
        if case["method"].startswith("U+"):
            u=np.load(root/f"results/test_scores/u_{case['seed']}.npz")
            mu,su=case["u_normalization"];ms,ss=case["s_normalization"]
            scores=np.maximum((u["scores"]-mu)/su,case["fusion_weight"]*(scores-ms)/ss)
            event_ms+=float(u["event_latency_ms"]);training_seconds+=float(u["training_seconds"])
        for kind in TASKS:
            m=evaluate(data,scores,case["threshold"],kind)
            rows.append({**case,**m,"test_type":kind,"is_known":kind==case["train_type"],
                         "event_latency_ms":event_ms,"training_seconds":training_seconds,
                         "parameter_count":int(payload["parameter_count"])})
        for variant in TYPES:
            variant_rows.append({**case, **evaluate(data,scores,case["threshold"],variant),
                                 "test_variant":variant,"is_known":variant in members(case["train_type"])})
        metadata.append({"case_index":index,"fit_id":case["fit_id"],"case_signature":signature(case)})
    pd.DataFrame(rows).to_csv(root/"results/test_per_setting.csv",index=False)
    pd.DataFrame(variant_rows).to_csv(root/"results/test_per_variant.csv",index=False)
    save_json(root/"results/test_case_index.json",metadata)
    print("TEST_COMPLETE",len(rows),"setting rows",flush=True)


def environment(root):
    versions={}
    for name in ["torch","numpy","pandas","scipy","scikit-learn","matplotlib","joblib","nbformat","nbconvert"]:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]="not installed"
    save_json(Path(root)/"environment/actual_versions.json",{"python":platform.python_version(),"platform":platform.platform(),
              "packages":versions,"worker_cap":16,"threads":{k:os.getenv(k) for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]}})


def smoke(root):
    root=Path(root);f=load(root);bank=bank_for(root)
    condition=f["conditions"][0];rep=condition["repeats"][0]
    train=bank.dataset(f["roles"]["train"],set(rep["train_keys"]),members(condition["train_type"]))
    val=bank.dataset(f["roles"]["validation"],set(rep["validation_keys"]),members(condition["train_type"]))
    results=[]
    for family in FAMILIES:
        config=copy.deepcopy(next(c for c in f["registry"] if c["family"]==family))
        if family!="RF":config.update(epochs=2,pretrain_epochs=2)
        model=Fitted(config,66).fit(train,root/"results/smoke")
        scores=model.score(val.x);threshold=exact_threshold(val.take(val.y==0),scores[val.y==0])
        restored=pickle.loads(cloudpickle.dumps(model))
        assert np.array_equal(scores,restored.score(val.x))
        results.append({"family":family,**evaluate(val,scores,threshold,condition["train_type"])})
    save_json(root/"results/smoke_test.json",results)
    print("SMOKE_PASS",len(results),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["freeze","baseline","search","select","test","smoke","environment"])
    parser.add_argument("--root",default=".");parser.add_argument("--workers",type=int,default=16)
    args=parser.parse_args()
    if args.action=="freeze":print(freeze(args.root)["signature"])
    elif args.action in ["baseline","search","select","test"]:
        {"baseline":baseline,"search":search,"select":freeze_selection,"test":evaluate_test}[args.action](args.root,args.workers)
    else:{"smoke":smoke,"environment":environment}[args.action](args.root)
