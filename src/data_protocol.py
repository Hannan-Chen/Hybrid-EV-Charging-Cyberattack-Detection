"""Immutable, source-grouped data and label access for supervised CMA experiments."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

TYPES = [f"Type-{i}" for i in range(1, 7)] + [f"Type-{i}-ta" for i in range(1, 7)]
ORDER = [0, 4, 2, 3, 1]
FEATURES = ["d_kWh", "duration_h", "st_offset_s", "et_offset_s", "rate_kW"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2), encoding="utf-8")
    temporary.replace(path)


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    mechanism: np.ndarray
    source: np.ndarray
    state: np.ndarray
    setting: np.ndarray
    update: np.ndarray
    key: np.ndarray

    def take(self, indices):
        return Dataset(**{k: getattr(self, k)[indices] for k in self.__dataclass_fields__})


class Bank:
    def __init__(self, root):
        self.root = Path(root)
        p = self.root / "data/processed"
        self.pool = joblib.load(p / "fixed_multi_input_attack_pool_seed66.joblib")
        self.states = pd.read_csv(p / "multi_input_state_catalog.csv")
        self.plan = json.loads((p / "multi_input_source_split_plan.json").read_text())
        self.source = np.asarray(self.pool["state_source_ids"], dtype=int)
        self.update = np.asarray(self.pool["state_event_types"]) == "user_update"
        self.profile = np.asarray(self.pool["normal_profile_features"], dtype=float)[:, ORDER]
        original = np.asarray(self.pool["normal_transition_features"], dtype=float)
        self.normal = np.column_stack([self.profile, original[:, np.array(ORDER) + 5], self.update])
        self.attack = {}
        self.eligible = {}
        for kind in TYPES:
            a = np.asarray(self.pool["attack_transition_features"][kind], dtype=float)
            self.attack[kind] = np.column_stack([a[:, ORDER], a[:, np.array(ORDER) + 5], a[:, 10]])
            self.eligible[kind] = np.asarray(self.pool["attack_state_indices"][kind], dtype=int)
        assert len(self.states) == len(self.normal) == 1264
        assert self.update.sum() == 51
        assert np.allclose(self.profile, self.states[FEATURES], rtol=2e-6, atol=0.02)
        assert np.all(self.normal[:, 10] == self.update)
        assert np.isfinite(self.normal).all()
        assert np.all(self.normal[~self.update, 5:10] == 0)
        for kind in TYPES:
            assert np.all(self.attack[kind][self.eligible[kind], 10] == 1)
            assert np.isfinite(self.attack[kind][self.eligible[kind]]).all()
        self.update_sources = set(self.source[self.update])
        self.source_ids = sorted(set(self.source))

    def available_keys(self, sources, fraction=1.0, subset_seed=66, types=None):
        allowed = set()
        records = []
        for index, kind in enumerate(TYPES):
            if kind not in (types or TYPES):
                continue
            eligible = self.eligible[kind]
            eligible = eligible[np.isin(self.source[eligible], sources)]
            groups = np.unique(self.source[eligible])
            order = np.random.default_rng(subset_seed + 1009 * index).permutation(groups)
            selected = order[:max(1, int(np.ceil(len(order) * fraction)))]
            chosen = eligible[np.isin(self.source[eligible], selected)]
            for state in chosen:
                key = f"{kind}:{state}"
                allowed.add(key)
                records.append({"key": key, "setting": kind, "state_id": int(state),
                                "source_id": int(self.source[state]), "fraction": fraction,
                                "subset_seed": subset_seed})
        return allowed, records

    def dataset(self, sources, allowed=None, types=None, normal_only=False):
        sources = list(sources)
        ids = np.flatnonzero(np.isin(self.source, sources))
        pieces = [(self.normal[ids], np.zeros(len(ids), int), np.zeros(len(ids), int),
                   self.source[ids], ids, np.full(len(ids), "Normal", object),
                   self.update[ids], np.array([f"N:{i}" for i in ids], object))]
        if not normal_only:
            for kind in (types or TYPES):
                state = self.eligible[kind]
                state = state[np.isin(self.source[state], sources)]
                if allowed is not None:
                    state = np.array([i for i in state if f"{kind}:{i}" in allowed], int)
                keys = np.array([f"{kind}:{i}" for i in state], object)
                mechanism = int(kind.split("-")[1])
                pieces.append((self.attack[kind][state], np.ones(len(state), int),
                               np.full(len(state), mechanism), self.source[state], state,
                               np.full(len(state), kind, object), np.zeros(len(state), bool), keys))
        return Dataset(**{k: np.concatenate([part[i] for part in pieces])
                          for i, k in enumerate(Dataset.__dataclass_fields__)})

    def strata(self, sources):
        return np.array([int(s in self.update_sources) for s in sources])

    def folds(self, sources, seed, dataset=None, types=None, fallback=False):
        sources = np.asarray(sorted(sources), int)
        attempts = [5, 3] if fallback else [5]
        for count in attempts:
            for attempt in range(20):
                splitter = StratifiedKFold(count, shuffle=True, random_state=seed + attempt * 10007)
                folds = []
                for train, val in splitter.split(sources, self.strata(sources)):
                    a, b = sources[train], sources[val]
                    if not set(b) & self.update_sources or not set(a) & self.update_sources:
                        break
                    if dataset is not None:
                        ok = all(np.any((dataset.setting == k) & np.isin(dataset.source, b))
                                 and np.any((dataset.setting == k) & np.isin(dataset.source, a))
                                 for k in (types or TYPES))
                        if not ok:
                            break
                    folds.append({"fold": len(folds), "fit_sources": a.tolist(), "eval_sources": b.tolist()})
                if len(folds) == count:
                    return folds
        raise ValueError("Insufficient independent labeled groups/updates for predefined folds")

    def reserve_normal_calibration(self, sources, seed):
        fit, cal = train_test_split(sorted(sources), test_size=0.2, random_state=seed,
                                    stratify=self.strata(sorted(sources)))
        assert set(fit).isdisjoint(cal)
        assert set(fit) & self.update_sources and set(cal) & self.update_sources
        return sorted(fit), sorted(cal)

    def audit(self):
        rows = []
        sets = []
        for role in ["final_train_source_ids", "calibration_source_ids", "test_source_ids"]:
            sources = self.plan[role]
            data = self.dataset(sources)
            normal = data.y == 0
            rows.append({"role": role, "sources": len(sources), "normal_states": int(normal.sum()),
                         "updates": int(data.update.sum()), "attacks": int(data.y.sum())})
            sets.append(set(sources))
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert len(set.union(*sets)) == 1213
        assert rows[-1]["attacks"] == 1505
        return rows


def sample_weights(data, types=None, update_mass=0.1):
    weights = np.zeros(len(data.y), float)
    groups = [(data.y == 0) & ~data.update, data.update]
    masses = [0.5 - update_mass, update_mass] if data.update.any() else [0.5, 0.0]
    present = types or sorted(set(data.setting[data.y == 1]))
    for kind in present:
        groups.append(data.setting == kind)
        masses.append(0.5 / len(present))
    for mask, mass in zip(groups, masses):
        if not mask.any():
            continue
        sources, counts = np.unique(data.source[mask], return_counts=True)
        count_map = dict(zip(sources, counts))
        weights[mask] = [mass / len(sources) / count_map[s] for s in data.source[mask]]
    assert weights.sum() > 0
    return weights * len(weights) / weights.sum()


def normal_threshold(data, scores):
    scores = np.asarray(scores, float)
    assert np.isfinite(scores).all()
    normal = data.y == 0
    assert normal.any()
    def cutoff(values, fpr):
        allowed = int(np.floor(fpr * len(values) + 1e-10))
        ordered = np.sort(values)
        return float(np.nextafter(ordered[len(values) - allowed - 1], np.inf))
    threshold = cutoff(scores[normal], 0.05)
    verified = bool(data.update.any())
    if verified:
        threshold = max(threshold, cutoff(scores[data.update], 0.10))
    prediction = scores >= threshold
    tnr = float((~prediction[normal]).mean())
    update_tnr = float((~prediction[data.update]).mean()) if verified else None
    assert tnr >= 0.95 - 1e-12
    assert not verified or update_tnr >= 0.90 - 1e-12
    return {"threshold": threshold, "normal_tnr": tnr, "update_tnr": update_tnr,
            "update_constraint_verified": verified, "calibration_normals": int(normal.sum()),
            "calibration_updates": int(data.update.sum())}


def metrics(data, scores, threshold, types=None):
    from sklearn.metrics import average_precision_score, roc_auc_score
    scores = np.asarray(scores, dtype=np.float64)
    prediction = scores >= threshold
    normal = data.y == 0
    def counts(mask):
        y, p = data.y[mask], prediction[mask]
        tn, fp = int(((y == 0) & ~p).sum()), int(((y == 0) & p).sum())
        fn, tp = int(((y == 1) & ~p).sum()), int(((y == 1) & p).sum())
        return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                "f1": 2 * tp / max(2 * tp + fp + fn, 1), "fpr": fp / max(fp + tn, 1)}
    result = counts(np.ones(len(data.y), bool))
    per = []
    for kind in (types or TYPES):
        attack = data.setting == kind
        if not attack.any():
            continue
        mask = normal | attack
        row = {"setting": kind, "attack_count": int(attack.sum()), **counts(mask)}
        row["pr_auc"] = float(average_precision_score(data.y[mask], np.asarray(scores)[mask]))
        row["roc_auc"] = float(roc_auc_score(data.y[mask], np.asarray(scores)[mask]))
        per.append(row)
    result.update(macro_f1=float(np.mean([r["f1"] for r in per])) if per else 0.0,
                  macro_recall=float(np.mean([r["recall"] for r in per])) if per else 0.0,
                  worst_recall=min((r["recall"] for r in per), default=0.0),
                  update_tnr=float((~prediction[data.update]).mean()) if data.update.any() else None,
                  update_fp=int(prediction[data.update].sum()), updates=int(data.update.sum()),
                  change_only_f1=counts(data.x[:, 10] == 1)["f1"])
    return result, per
