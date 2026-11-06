"""Independent NumPy/count-based verification, usable locally without PyTorch."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from data_protocol import Bank, digest, save_json
from protocol import signature


def verify(root=ROOT):
    root=Path(root);checks=0
    def check(value,message):
        nonlocal checks
        checks+=1
        if not bool(value):raise AssertionError(message)
    def close(a,b,message):check(np.isclose(float(a),float(b),rtol=2e-10,atol=2e-10),message)
    design=json.loads((root/"results/frozen_design.json").read_text())
    check(signature({k:v for k,v in design.items() if k!="signature"})==design["signature"],"protocol signature")
    public_path=root/"data/public_release_provenance.json"
    public=json.loads(public_path.read_text()) if public_path.exists() else {}
    for name,expected in design["input_code_sha256"].items():
        omitted=public.get("omitted_original_inputs",{})
        changed=public.get("original_inputs_transformed",{})
        if name in omitted:
            check(omitted[name]==expected and not (root/name).exists(),"documented raw-data omission: "+name)
        elif name in changed:
            entry=changed[name]
            check(entry["original_sha256"]==expected,"original provenance: "+name)
            check(digest(root/name)==entry["released_sha256"],"public transformed input: "+name)
        else:
            check(digest(root/name)==expected,name)
    selection=json.loads((root/"results/final_selection_freeze.json").read_text())
    check(signature({k:v for k,v in selection.items() if k!="signature"})==selection["signature"],"selection signature")
    check(selection["protocol_signature"]==design["signature"],"selection protocol binding")
    for name,expected in selection["selection_files"].items():check(digest(root/name)==expected,name)
    roles=design["roles"];bank=Bank(root)
    for a in roles:
        for b in roles:
            if a!=b:check(not(set(roles[a])&set(roles[b])),"source isolation")
    normal=bank.dataset(roles["validation"],normal_only=True)
    test=bank.dataset(roles["test"])
    u_cache={s:joblib.load(root/f"results/cache/u_{s}.joblib") for s in design["seeds"]}
    validation_cache={}
    def variants(kind):
        return ["Type-"+kind.split("-")[1],"Type-"+kind.split("-")[1]+"-ta"]
    def validation_data(condition):
        if condition["id"] not in validation_cache:
            r=condition["repeats"][0]
            validation_cache[condition["id"]]=bank.dataset(roles["validation"],set(r["validation_keys"]),variants(condition["train_type"]))
        return validation_cache[condition["id"]]
    def cutoff(scores):
        overall=np.sort(scores)[len(scores)-int(np.floor(.05*len(scores)+1e-10))-1]
        u=scores[normal.update];edit=np.sort(u)[len(u)-int(np.floor(.1*len(u)+1e-10))-1]
        return float(np.nextafter(max(overall,edit),np.inf))
    def metrics(y,update,setting,scores,threshold,kind):
        pred=scores>=threshold;n=y==0
        grouped=kind.startswith("Mechanism-")
        a=np.isin(setting,variants(kind)) if grouped else setting==kind
        u=update&n
        tp=int(np.count_nonzero(pred&a));fn=int(np.count_nonzero(~pred&a))
        fp=int(np.count_nonzero(pred&n));tn=int(np.count_nonzero(~pred&n))
        uf=int(np.count_nonzero(pred&u));ut=int(np.count_nonzero(~pred&u))
        ap=2*tp/max(2*tp+uf+fn,1);np_=2*ut/max(2*ut+uf+fn,1)
        result={"tp":tp,"fn":fn,"fp":fp,"tn":tn,"update_fp":uf,"updates":uf+ut,
                "recall":tp/max(tp+fn,1),"normal_fpr":fp/max(tn+fp,1),"update_fpr":uf/max(uf+ut,1),
                "change_class_macro_f1":(ap+np_)/2,"change_attack_f1":ap,"change_normal_f1":np_,
                "change_balanced_accuracy":.5*(tp/max(tp+fn,1)+ut/max(ut+uf,1)),
                "legacy_attack_f1":2*tp/max(2*tp+fp+fn,1)}
        if grouped:
            per=[metrics(y,update,setting,scores,threshold,v) for v in variants(kind)]
            for field in ["change_class_macro_f1","change_attack_f1","change_normal_f1","change_balanced_accuracy","recall","legacy_attack_f1"]:
                result["pooled_"+field]=result[field]
                result[field]=sum(r[field] for r in per)/2
            result["minimum_variant_recall"]=min(r["recall"] for r in per)
        return result
    failures=[];fit_count=0
    for condition in design["conditions"]:
        for rep in condition["repeats"]:
            keys=rep["train_keys"];states=[int(k.rsplit(":",1)[1]) for k in keys]
            vs=[int(k.rsplit(":",1)[1]) for k in rep["validation_keys"]]
            check(len(set(bank.source[states]))==len(keys),"unique attack training sources")
            check(len(vs)==20 and len(set(bank.source[vs]))==20,"unique validation attack sources")
            check(set(bank.source[states])<=set(roles["train"]),"train label access")
            check(set(bank.source[vs])<=set(roles["validation"]),"validation label access")
            allowed=set(variants(condition["train_type"]))
            check(all(k.split(":")[0] in allowed for k in keys+rep["validation_keys"]),"one mechanism access")
            val_counts=[sum(k.startswith(v+":") for k in rep["validation_keys"]) for v in variants(condition["train_type"])]
            train_counts=[sum(k.startswith(v+":") for k in keys) for v in variants(condition["train_type"])]
            check(val_counts==[10,10],"balanced validation variants")
            if condition["budget"]!="All":check(abs(train_counts[0]-train_counts[1])<=1,"balanced finite training budget")
            else:
                eligible=set()
                for v in allowed:
                    for state in bank.eligible[v]:
                        if int(bank.source[state]) in set(roles["train"]):eligible.add(int(bank.source[state]))
                check(set(bank.source[states])==eligible,"All covers eligible source union")
            if condition["budget"]!="All":check(len(keys)==condition["budget"],"budget count")
            for other in design["conditions"]:
                if other["train_type"]==condition["train_type"] and (other["budget"]=="All" or
                     condition["budget"]!="All" and other["budget"]>condition["budget"]):
                    bigger=next(r for r in other["repeats"] if r["seed"]==rep["seed"])
                    check(set(keys)<=set(bigger["train_keys"]),"nested budget")
        folder=root/"results/development"/condition["id"]
        table=pd.read_csv(folder/"candidate_seed_results.csv",float_precision="round_trip")
        winners=json.loads((folder/"selection.json").read_text())["winners"]
        check(len(table)==205,"all seven systems' candidate repetitions")
        # Reconstruct ranking directly from per-repeat metrics, not saved ranking columns.
        for method,g in table.groupby("method"):
            rank=[]
            for cid,group in g.groupby("config_id"):
                check(set(group.seed)==set(design["seeds"]) and len(group)==5,"paired seeds")
                success=bool((group.status=="ok").all())
                x=group.change_class_macro_f1.to_numpy()
                cfg=next((c for c in design["registry"] if c["id"]==cid.split("_w")[0]),{"complexity":0})
                rank.append((not success,-(x.mean()-.5*x.std(ddof=1)),group.update_fpr.mean(),group.normal_fpr.mean(),cfg["complexity"],cid))
            expected=sorted(rank)[0][-1]
            actual=next(w["config_id"] for w in winners if w["method"]==method)
            check(expected==actual,"validation-only family selection")
        for row in table.to_dict("records"):
            if row["status"]!="ok":failures.append(row);continue
            seed=int(row["seed"]);method=row["method"]
            if method=="U":
                p=u_cache[seed]
                scores=p["validation"][condition["train_type"]]["scores"]
                rep=next(r for r in condition["repeats"] if r["seed"]==seed)
                data=validation_data(condition)
            else:
                p=joblib.load(root/"results/cache/development"/(row["fit_id"]+".joblib"))
                scores=p["scores"]
                rep=next(r for r in condition["repeats"] if r["seed"]==seed)
                data=validation_data(condition)
                check(set(p["fit_attack_keys"])==set(rep["train_keys"]),"fit label manifest")
                check(set(p["validation_attack_keys"])==set(rep["validation_keys"]),"validation label manifest")
                check(np.array_equal(p["keys"],data.key),"validation record identity")
                if method.startswith("U+"):
                    u=u_cache[seed]
                    med,scale=p["normalization"];um,us=u["normalization"]
                    scores=np.maximum((u["validation"][condition["train_type"]]["scores"]-um)/us,
                                      row["fusion_weight"]*(scores-med)/scale)
                else:fit_count+=1
            check(np.isfinite(scores).all(),"finite validation scores")
            close(row["threshold"],cutoff(scores[data.y==0]),"independent normal-only threshold")
            values=metrics(data.y,data.update,data.setting,scores,row["threshold"],condition["train_type"])
            for name,value in values.items():close(row[name],value,"validation "+name)
    result=pd.read_csv(root/"results/test_per_setting.csv",float_precision="round_trip",low_memory=False)
    variant_results=pd.read_csv(root/"results/test_per_variant.csv",float_precision="round_trip",low_memory=False)
    caseindex=json.loads((root/"results/test_case_index.json").read_text())
    check(len(caseindex)==len(selection["cases"]),"test case completeness")
    check(len(result)==6*len(selection["cases"]),"all six-mechanism test rows")
    check(len(variant_results)==12*len(selection["cases"]),"all twelve-variant test rows")
    for case in selection["cases"]:
        path=root/"results/test_scores"/(case["fit_id"]+".npz")
        p=np.load(path);scores=p["scores"]
        check(np.array_equal(p["keys"],test.key.astype(str)),"final test identity")
        check(str(p["model_sha256"])==case["model_sha256"],"frozen evaluated weights")
        model_name="results/cache/models/"+case["fit_id"]+".pkl"
        model_change=public.get("original_inputs_transformed",{}).get(model_name)
        if model_change:
            check(model_change["original_sha256"]==case["model_sha256"],"original frozen weight provenance")
            check(digest(root/model_name)==model_change["released_sha256"],"public weights with redacted source-location metadata")
        else:
            check(digest(root/model_name)==case["model_sha256"],"downloaded weights")
        if case["method"].startswith("U+"):
            u=np.load(root/f"results/test_scores/u_{case['seed']}.npz")
            um,us=case["u_normalization"];sm,ss=case["s_normalization"]
            scores=np.maximum((u["scores"]-um)/us,case["fusion_weight"]*(scores-sm)/ss)
        check(np.isfinite(scores).all(),"finite test scores")
        g=result[(result.condition_id==case["condition_id"])&(result.method==case["method"])&(result.seed==case["seed"])]
        check(len(g)==6,"six mechanism evaluations per frozen case")
        for row in g.to_dict("records"):
            close(row["threshold"],case["threshold"],"frozen threshold at test")
            for name,value in metrics(test.y,test.update,test.setting,scores,case["threshold"],row["test_type"]).items():
                close(row[name],value,"test "+name)
        vg=variant_results[(variant_results.condition_id==case["condition_id"])&
                           (variant_results.method==case["method"])&(variant_results.seed==case["seed"])]
        check(len(vg)==12,"twelve variant evaluations per frozen case")
        for row in vg.to_dict("records"):
            close(row["threshold"],case["threshold"],"frozen variant threshold")
            for name,value in metrics(test.y,test.update,test.setting,scores,case["threshold"],row["test_variant"]).items():
                close(row[name],value,"variant "+name)
    books=list((root/"juno_executed").glob("*.ipynb"))
    check(len(books)>=4,"executed notebooks available")
    for path in books:
        doc=json.loads(path.read_text(encoding="utf-8"))
        for cell in doc["cells"]:
            if cell["cell_type"]=="code":
                check(cell["execution_count"] is not None,"notebook executed")
                check(not any(o.get("output_type")=="error" for o in cell.get("outputs",[])),"notebook no error")
    for pdf in (root/"figures").glob("*.pdf"):
        check(pdf.stat().st_size>1000,"nonempty figure")
        check(pdf.with_suffix(".svg").exists(),"vector SVG pair")
    report={"status":"passed","assertions":checks,"development_base_fits":fit_count,
            "recorded_fit_failures":len(failures),"frozen_test_cases":len(selection["cases"]),
            "setting_rows":len(result),"variant_rows":len(variant_results),"protocol_signature":design["signature"],
            "selection_signature":selection["signature"],
            "public_input_transformations_verified":bool(public),
            "source_location_metadata_redacted_models":public.get("source_location_metadata_redacted_models",0),
            "scope":"independent source/label/threshold/selection/count audit with documented public de-identification; not proof of real-world generalization"}
    save_json(root/"reports/local_verification.json",report)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=="__main__":
    verify(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT)
