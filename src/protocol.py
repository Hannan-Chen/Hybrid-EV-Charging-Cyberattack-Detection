"""Source-fixed, label-budgeted protocol; no fitting or test score access."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_protocol import Bank, TYPES, digest, save_json

SEEDS = [66, 166, 266, 366, 466]
BUDGETS = [2, 5, 10, 20, 50, "All"]
FAMILIES = ["RF", "SDTN", "DeepSAD", "SS"]
METHODS = ["U", "RF", "SDTN", "DeepSAD", "SS", "U+RF", "U+SDTN"]
TASKS = [f"Mechanism-{i}" for i in range(1, 7)]
U_CONFIG = {"family": "Normal-only Dual-Branch", "protocol": "transition", "params": {
    "hidden": [64, 32], "latent": 4, "mask_rate": .15, "gamma": "scale",
    "transition_weight": .15, "nu": .05, "epochs": 180, "transition_epochs": 160}}


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def registry():
    rows = []
    for depth, leaf in itertools.product([8, None], [1, 5]):
        rows.append({"id": f"rf{len(rows)}", "family": "RF", "depth": depth, "leaf": leaf,
                     "trees": 128, "complexity": 128 * 2 ** (depth or 12) / leaf})
    for family in ["SDTN", "DeepSAD", "SS"]:
        for width, wd in itertools.product([32, 64], [1e-5, 1e-3]):
            extra = [.1, 1.] if family in ["DeepSAD", "SS"] else [0.]
            for value in extra:
                rows.append({"id": f"{family.lower()}{len(rows)}", "family": family,
                             "width": width, "embedding": 8 if width == 32 else 16,
                             "weight_decay": wd, "epochs": 160, "pretrain_epochs": 100,
                             "lr": .001, "mask_rate": .15, "extra": value,
                             "complexity": width ** 2})
    return rows


def source_roles(bank):
    test = sorted(bank.plan["test_source_ids"])
    available = sorted(set(bank.source_ids) - set(test))
    rng = np.random.default_rng(20261026)
    updates = rng.permutation([s for s in available if s in bank.update_sources]).tolist()
    other = rng.permutation([s for s in available if s not in bank.update_sources]).tolist()
    return {"train": sorted(updates[:22] + other[:584]),
            "validation": sorted(updates[22:] + other[584:]), "test": test}


def canonical_keys(bank, sources, kind):
    eligible = bank.eligible[kind]
    eligible = eligible[np.isin(bank.source[eligible], sources)]
    by_source = {}
    for state in sorted(eligible):
        by_source.setdefault(int(bank.source[state]), f"{kind}:{int(state)}")
    return [by_source[s] for s in sorted(by_source)]


def members(kind):
    if kind in TASKS:
        index = kind.split("-")[1]
        return [f"Type-{index}", f"Type-{index}-ta"]
    assert kind in TYPES, kind
    return [kind]


def paired_bank(bank, sources, kind, seed):
    """One derivative per source, balanced nested prefix, all union sources at All."""
    variants = members(kind)
    assert len(variants) == 2
    maps = [{int(bank.source[int(k.rsplit(":", 1)[1])]): k
             for k in canonical_keys(bank, sources, variant)} for variant in variants]
    sets = [set(m) for m in maps]
    rng = np.random.default_rng(seed)
    shared = rng.permutation(sorted(sets[0] & sets[1])).tolist()
    exclusive = [rng.permutation(sorted(sets[j]-sets[1-j])).tolist() for j in range(2)]
    quota = min(len(sets[0]), len(sets[1]), len(sets[0] | sets[1])//2)
    chosen = [ex[:quota] for ex in exclusive]
    used = set(chosen[0]+chosen[1])
    cursor = 0
    for j in range(2):
        missing = quota-len(chosen[j])
        selected = shared[cursor:cursor+missing]
        assert len(selected) == missing
        chosen[j].extend(selected);used.update(selected);cursor += missing
    queues = [rng.permutation([maps[j][source] for source in chosen[j]]).tolist() for j in range(2)]
    first = int(rng.integers(2))
    ordered = [queues[j][i] for i in range(quota) for j in [first, 1-first]]
    # Balanced finite budgets do not require dropping unique sources from the All endpoint.
    for source in rng.permutation(sorted((sets[0] | sets[1])-used)):
        options = [j for j in range(2) if int(source) in maps[j]]
        ordered.append(maps[int(rng.choice(options))][int(source)])
    assert len(ordered) == len(sets[0] | sets[1])
    assert len({int(bank.source[int(k.rsplit(":",1)[1])]) for k in ordered}) == len(ordered)
    return ordered


def freeze(root):
    root = Path(root)
    bank = Bank(root)
    roles = source_roles(bank)
    assert [len(roles[r]) for r in roles] == [606, 364, 243]
    assert not any(set(roles[a]) & set(roles[b]) for a, b in itertools.combinations(roles, 2))
    assert bank.pool["control_params"] == {"Catt": .15, "Chav": 30., "Pl": 24., "dt": 5}
    audit, conditions = [], []
    for role, sources in roles.items():
        data = bank.dataset(sources, normal_only=True)
        audit.append({"role": role, "sources": len(sources), "normal_states": len(data.y),
                      "updates": int(data.update.sum()), "update_sources": len(set(data.source[data.update]))})
    assert [(r["normal_states"], r["updates"]) for r in audit] == [(630,24),(382,18),(252,9)]
    for ti, kind in enumerate(TASKS):
        available = paired_bank(bank, roles["train"], kind, 66 + 1009*ti)
        val = paired_bank(bank, roles["validation"], kind, 66 + 1009*ti)[:20]
        assert len(val) == 20 and len(available) >= 50
        for budget in BUDGETS:
            repeats = []
            for seed in SEEDS:
                ordered = paired_bank(bank, roles["train"], kind, seed + 1009*ti)
                train = ordered if budget == "All" else ordered[:budget]
                repeats.append({"seed": seed, "train_keys": train, "validation_keys": val,
                                "train_variant_counts": {v:sum(k.startswith(v+":") for k in train) for v in members(kind)},
                                "validation_variant_counts": {v:sum(k.startswith(v+":") for k in val) for v in members(kind)},
                                "attack_train_count": len(train), "attack_label_cost": len(train)+20})
            conditions.append({"id": kind.replace("Mechanism-", "m") + "_n" + str(budget),
                               "train_type": kind, "train_variants": members(kind), "budget": budget, "bank_size": len(available),
                               "repeats": repeats})
    files = {p.relative_to(root).as_posix(): digest(p) for folder in ["src", "data", "reference", "tests", "environment"]
             for p in sorted((root / folder).rglob("*")) if p.is_file() and "__pycache__" not in p.parts
             and p.name != "actual_versions.json"}
    design = {"version": 2, "tasks": TASKS, "roles": roles, "audit": audit, "seeds": SEEDS, "budgets": BUDGETS,
              "conditions": conditions, "registry": registry(), "u_config": U_CONFIG,
              "methods": METHODS, "input_code_sha256": files, "attack_controls": bank.pool["control_params"],
              "threshold": "lowest exact >= boundary accepting at least .95 normals and .90 real updates; normal validation only",
              "selection": "all fits successful; mean change binary-class-macro F1 minus .5 sample SD; lower update FPR, overall FPR, complexity, id",
              "score_normalization": "training-normal median/IQR; scale floor=max(1e-6,.01*SD)",
              "parallel_weights": [.5, 1.], "bootstrap_seed": 20261104, "bootstrap_replicates": 1000,
              "cross_validation": False, "refit_train_validation": False, "attacks_regenerated": False,
              "benign_edits_simulated": False, "test_scope": "reused simulated benchmark, not external validation",
              "normal_weight_masses": {"activation": .4, "real_update": .1, "attack": .5},
              "ablation_budgets": [5, 20, "All"], "ablations": ["no_reconstruction", "no_retention"],
              "architecture_scope": "historically motivated architectures; no claim that mechanisms were unknown to researchers"}
    design["mechanism_semantics"] = "Type-i and Type-i-ta jointly fitted and selected; equal variant-weighted metrics, not post-hoc pooling"
    design["budget_semantics"] = "N total attack source records per mechanism, unique across variants; finite budgets balanced within one; All covers union with imbalance reported"
    design["signature"] = signature(design)
    path = root / "results/frozen_design.json"
    if path.exists():
        assert json.loads(path.read_text())["signature"] == design["signature"], "Frozen protocol changed"
    else:
        save_json(path, design)
    pd.DataFrame(audit).to_csv(root / "results/data_audit.csv", index=False)
    return design


def load(root):
    root = Path(root)
    d = json.loads((root / "results/frozen_design.json").read_text())
    assert signature({k: v for k, v in d.items() if k != "signature"}) == d["signature"]
    for name, expected in d["input_code_sha256"].items():
        assert digest(root / name) == expected, name
    return d


def exact_threshold(normal_data, scores):
    scores = np.asarray(scores, float)
    assert np.all(normal_data.y == 0) and np.isfinite(scores).all()
    def boundary(values, rate):
        allowed = int(np.floor(rate * len(values) + 1e-10))
        return np.nextafter(np.sort(values)[len(values)-allowed-1], np.inf)
    threshold = float(max(boundary(scores, .05), boundary(scores[normal_data.update], .10)))
    pred = scores >= threshold
    assert pred.sum() <= np.floor(.05*len(pred)+1e-10)
    assert pred[normal_data.update].sum() <= np.floor(.1*normal_data.update.sum()+1e-10)
    return threshold


def evaluate(data, scores, threshold, kind):
    p = np.asarray(scores, float) >= threshold
    variants = members(kind)
    attack = np.isin(data.setting, variants)
    normal = data.y == 0
    update = data.update & normal
    tp, fn = int(p[attack].sum()), int((~p[attack]).sum())
    fp, tn = int(p[normal].sum()), int((~p[normal]).sum())
    ufp, utn = int(p[update].sum()), int((~p[update]).sum())
    pos = 2*tp/max(2*tp+ufp+fn, 1)
    neg = 2*utn/max(2*utn+ufp+fn, 1)
    activation = normal & ~update
    result = {"change_class_macro_f1": (pos+neg)/2, "change_attack_f1": pos,
            "change_normal_f1": neg, "change_balanced_accuracy": .5*(tp/max(tp+fn,1)+utn/max(utn+ufp,1)),
            "recall": tp/max(tp+fn,1), "update_fp": ufp, "updates": ufp+utn,
            "update_fpr": ufp/max(ufp+utn,1), "normal_fpr": fp/max(fp+tn,1),
            "activation_fpr": float(p[activation].mean()), "legacy_attack_f1": 2*tp/max(2*tp+fp+fn,1),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "constant_prediction": bool(p.min() == p.max()), "threshold": threshold}
    if len(variants) == 2:
        per = [evaluate(data, scores, threshold, variant) for variant in variants]
        for name in ["change_class_macro_f1", "change_attack_f1", "change_normal_f1",
                     "change_balanced_accuracy", "recall", "legacy_attack_f1"]:
            result["pooled_"+name] = result[name]
            result[name] = float(np.mean([row[name] for row in per]))
        result["minimum_variant_recall"] = min(row["recall"] for row in per)
    return result
