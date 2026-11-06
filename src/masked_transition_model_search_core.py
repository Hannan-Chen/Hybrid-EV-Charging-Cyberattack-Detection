"""Leakage-controlled masked and transition-aware session-model search."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


SEED = 66
N_FOLDS = 5
MAX_WORKERS = min(16, max(1, int(os.environ.get("SESSION_MODEL_MAX_WORKERS", "16"))))
OVERALL_TNR_TARGET = 0.95
UPDATE_TNR_TARGET = 0.90
STABILITY_PENALTY = 0.50
ATTACK_TYPES = [f"Type-{i}" for i in range(1, 7)] + [f"Type-{i}-ta" for i in range(1, 7)]


def configure_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def source_mask(source_array, source_ids) -> np.ndarray:
    return np.isin(np.asarray(source_array, dtype=int), np.asarray(list(source_ids), dtype=int))


def empirical_percentile(reference, values) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    return np.searchsorted(reference, values, side="right") / max(len(reference), 1)


def continuous_standard_score(reference, values) -> np.ndarray:
    """Preserve the complete score ordering while putting heads on comparable scales."""
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    return (values - float(reference.mean())) / float(reference.std() + 1e-8)


def compact_transition(current, previous) -> np.ndarray:
    """Dimensionless changes with a direct charging interpretation."""
    current = np.asarray(current, dtype=float)
    previous = np.asarray(previous, dtype=float)
    eps = 1e-8
    previous_duration_s = np.maximum(np.abs(previous[:, 4]) * 3600.0, eps)
    return np.column_stack(
        [
            (current[:, 0] - previous[:, 0]) / np.maximum(np.abs(previous[:, 0]), eps),
            (current[:, 1] - previous[:, 1]) / np.maximum(np.abs(previous[:, 1]), eps),
            (current[:, 2] - previous[:, 2]) / previous_duration_s,
            (current[:, 3] - previous[:, 3]) / previous_duration_s,
            (current[:, 4] - previous[:, 4]) / np.maximum(np.abs(previous[:, 4]), eps),
        ]
    )


@dataclass
class DataBundle:
    root: Path
    states: pd.DataFrame
    pool: dict
    plan: dict
    profile: np.ndarray
    compact: np.ndarray
    is_change: np.ndarray
    source_ids: np.ndarray
    station_ids: np.ndarray
    attack_compact: dict
    candidate_signature: str = ""


def load_data(root: Path) -> DataBundle:
    root = Path(root).resolve()
    processed = root / "data" / "processed"
    states = pd.read_csv(processed / "multi_input_state_catalog.csv")
    pool = joblib.load(processed / "fixed_multi_input_attack_pool_seed66.joblib")
    plan = json.loads((processed / "multi_input_source_split_plan.json").read_text(encoding="utf-8"))
    raw = json.loads((root / "data" / "raw" / "caltech_sessions_2021.json").read_text(encoding="utf-8"))

    valid_records = [
        record for record in raw
        if (record.get("userInputs") or [])
        and float(record["userInputs"][0].get("kWhRequested", 0.0)) > 0
    ]
    if len(valid_records) != 1213:
        raise RuntimeError(f"Unexpected valid source count: {len(valid_records)}")
    source_station = np.asarray([str(record.get("stationID", "UNKNOWN")) for record in valid_records], dtype=object)

    profile = np.asarray(pool["normal_profile_features"], dtype=float)
    transition = np.asarray(pool["normal_transition_features"], dtype=float)
    previous = profile - transition[:, 5:10]
    compact = compact_transition(profile, previous)
    is_change = np.asarray(pool["state_event_types"], dtype=str) == "user_update"
    compact[~is_change] = 0.0
    state_sources = np.asarray(pool["state_source_ids"], dtype=int)
    station_ids = source_station[state_sources]

    attack_compact = {}
    for attack_type in ATTACK_TYPES:
        attacked = np.asarray(pool["attack_profile_features"][attack_type], dtype=float)
        attack_compact[attack_type] = compact_transition(attacked, profile)

    if len(profile) != 1264 or int(is_change.sum()) != 51:
        raise RuntimeError("Multi-input state audit failed")
    return DataBundle(root, states, pool, plan, profile, compact, is_change, state_sources, station_ids, attack_compact)


class StationShrinkTransformer:
    def __init__(self, kappa: float):
        self.kappa = float(kappa)

    def fit(self, x, stations):
        x = np.asarray(x, dtype=float)
        stations = np.asarray(stations, dtype=object)
        self.global_mean = x.mean(axis=0)
        self.global_var = x.var(axis=0) + 1e-6
        self.station_stats = {}
        for station in np.unique(stations):
            part = x[stations == station]
            n = len(part)
            weight = n / (n + self.kappa)
            mean = weight * part.mean(axis=0) + (1.0 - weight) * self.global_mean
            var = weight * (part.var(axis=0) + 1e-6) + (1.0 - weight) * self.global_var
            self.station_stats[str(station)] = (mean, var)
        return self

    def transform(self, x, stations):
        x = np.asarray(x, dtype=float)
        stations = np.asarray(stations, dtype=object)
        global_z = (x - self.global_mean) / np.sqrt(self.global_var)
        local_z = np.empty_like(x)
        for i, station in enumerate(stations):
            mean, var = self.station_stats.get(str(station), (self.global_mean, self.global_var))
            local_z[i] = (x[i] - mean) / np.sqrt(var)
        return np.column_stack([global_z, local_z])


class MaskedAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], latent: int):
        super().__init__()
        encoder = []
        previous = input_dim
        for width in hidden:
            encoder += [nn.Linear(previous, int(width)), nn.GELU()]
            previous = int(width)
        encoder.append(nn.Linear(previous, int(latent)))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        previous = int(latent)
        for width in reversed(hidden):
            decoder += [nn.Linear(previous, int(width)), nn.GELU()]
            previous = int(width)
        decoder.append(nn.Linear(previous, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def fit_masked_ae(x_train, hidden, latent, mask_rate, epochs, seed):
    configure_seed(seed)
    scaler = StandardScaler().fit(x_train)
    x = scaler.transform(x_train).astype(np.float32)
    model = MaskedAutoencoder(x.shape[1], list(hidden), int(latent))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    tensor = torch.from_numpy(x)
    generator = torch.Generator().manual_seed(int(seed) + 19)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tensor), batch_size=min(96, len(x)), shuffle=True, generator=generator
    )
    model.train()
    for _ in range(int(epochs)):
        for (batch,) in loader:
            mask = torch.rand(batch.shape, generator=generator) < float(mask_rate)
            corrupted = batch.masked_fill(mask, 0.0)
            reconstruction, _ = model(corrupted)
            masked_loss = ((reconstruction - batch) ** 2 * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
            full_loss = ((reconstruction - batch) ** 2).mean()
            loss = masked_loss + 0.10 * full_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return {"scaler": scaler, "model": model}


def ae_outputs(fit, x):
    scaled = fit["scaler"].transform(x).astype(np.float32)
    with torch.no_grad():
        reconstruction, latent = fit["model"](torch.from_numpy(scaled))
    residual = scaled - reconstruction.numpy()
    return scaled, reconstruction.numpy(), latent.numpy(), residual


class SVDDNet(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], embed_dim: int):
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden:
            layers += [nn.Linear(previous, int(width), bias=False), nn.Tanh()]
            previous = int(width)
        layers.append(nn.Linear(previous, int(embed_dim), bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_svdd(x_train, hidden, embed_dim, epochs, seed):
    configure_seed(seed)
    scaler = StandardScaler().fit(x_train)
    x = scaler.transform(x_train).astype(np.float32)
    tensor = torch.from_numpy(x)
    model = SVDDNet(x.shape[1], list(hidden), int(embed_dim))
    with torch.no_grad():
        center = model(tensor).mean(dim=0)
        center[(center.abs() < 0.05) & (center >= 0)] = 0.05
        center[(center.abs() < 0.05) & (center < 0)] = -0.05
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(int(seed) + 23)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tensor), batch_size=min(96, len(x)), shuffle=True, generator=generator
    )
    model.train()
    for _ in range(int(epochs)):
        for (batch,) in loader:
            embedding = model(batch)
            loss = ((embedding - center) ** 2).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return {"scaler": scaler, "model": model, "center": center}


def svdd_scores(fit, x):
    scaled = fit["scaler"].transform(x).astype(np.float32)
    with torch.no_grad():
        embedding = fit["model"](torch.from_numpy(scaled))
    return ((embedding - fit["center"]) ** 2).sum(dim=1).numpy()


def fit_transition_head(kind, x_train, params, seed):
    x_train = np.asarray(x_train, dtype=float)
    if len(x_train) < 8:
        raise RuntimeError("A transition head requires at least eight observed normal updates")
    if kind == "mahalanobis":
        scaler = StandardScaler().fit(x_train)
        model = LedoitWolf().fit(scaler.transform(x_train))
        raw = lambda x: model.mahalanobis(scaler.transform(x))
    elif kind == "ocsvm":
        scaler = StandardScaler().fit(x_train)
        model = OneClassSVM(kernel="rbf", nu=float(params.get("transition_nu", 0.10)), gamma="scale").fit(
            scaler.transform(x_train)
        )
        raw = lambda x: -model.decision_function(scaler.transform(x)).reshape(-1)
    elif kind == "svdd":
        model = fit_svdd(x_train, [12, 6], 3, int(params.get("transition_epochs", 180)), seed)
        raw = lambda x: svdd_scores(model, x)
    elif kind == "masked_ae":
        model = fit_masked_ae(x_train, [12, 6], 2, 0.25, int(params.get("transition_epochs", 180)), seed)
        _, _, latent_train, residual_train = ae_outputs(model, x_train)
        rep_train = np.column_stack([residual_train, latent_train])
        rep_scaler = StandardScaler().fit(rep_train)
        covariance = LedoitWolf().fit(rep_scaler.transform(rep_train))

        def raw(x):
            _, _, latent, residual = ae_outputs(model, x)
            return covariance.mahalanobis(rep_scaler.transform(np.column_stack([residual, latent])))
    else:
        raise ValueError(kind)
    reference = np.asarray(raw(x_train), dtype=float)
    return lambda x: continuous_standard_score(reference, raw(np.asarray(x, dtype=float)))


def fit_state_head(kind, x_train, params, seed):
    x_train = np.asarray(x_train, dtype=float)
    if kind == "ocsvm":
        scaler = StandardScaler().fit(x_train)
        model = OneClassSVM(
            kernel="rbf", nu=float(params.get("nu", 0.10)), gamma=params.get("gamma", 1.0)
        ).fit(scaler.transform(x_train))
        raw = lambda x: -model.decision_function(scaler.transform(x)).reshape(-1)
    elif kind == "masked_ae_ocsvm":
        ae = fit_masked_ae(
            x_train, params["hidden"], params["latent"], params["mask_rate"], params["epochs"], seed
        )
        _, _, latent_train, residual_train = ae_outputs(ae, x_train)
        rep_train = np.column_stack([residual_train, latent_train])
        rep_scaler = StandardScaler().fit(rep_train)
        model = OneClassSVM(
            kernel="rbf", nu=float(params.get("nu", 0.05)), gamma=params.get("gamma", "scale")
        ).fit(rep_scaler.transform(rep_train))

        def raw(x):
            _, _, latent, residual = ae_outputs(ae, x)
            rep = rep_scaler.transform(np.column_stack([residual, latent]))
            return -model.decision_function(rep).reshape(-1)
    elif kind == "masked_ae_energy":
        ae = fit_masked_ae(
            x_train, params["hidden"], params["latent"], params["mask_rate"], params["epochs"], seed
        )
        scaled_train, reconstruction_train, latent_train, _ = ae_outputs(ae, x_train)
        recon_reference = np.mean((scaled_train - reconstruction_train) ** 2, axis=1)
        latent_scaler = StandardScaler().fit(latent_train)
        latent_model = LedoitWolf().fit(latent_scaler.transform(latent_train))
        latent_reference = latent_model.mahalanobis(latent_scaler.transform(latent_train))
        alpha = float(params.get("energy_alpha", 0.5))

        def raw(x):
            scaled, reconstruction, latent, _ = ae_outputs(ae, x)
            recon = np.mean((scaled - reconstruction) ** 2, axis=1)
            latent_distance = latent_model.mahalanobis(latent_scaler.transform(latent))
            return alpha * empirical_percentile(recon_reference, recon) + (1.0 - alpha) * empirical_percentile(
                latent_reference, latent_distance
            )
    elif kind == "multiview_masked_ae":
        ae = fit_masked_ae(
            x_train, params["hidden"], params["latent"], params["mask_rate"], params["epochs"], seed
        )

        def view_reconstruction_score(x):
            scaled = ae["scaler"].transform(x).astype(np.float32)
            errors = []
            with torch.no_grad():
                for feature in range(scaled.shape[1]):
                    masked = scaled.copy()
                    masked[:, feature] = 0.0
                    reconstruction, _ = ae["model"](torch.from_numpy(masked))
                    errors.append((reconstruction.numpy()[:, feature] - scaled[:, feature]) ** 2)
            return np.mean(np.column_stack(errors), axis=1)

        _, _, latent_train, _ = ae_outputs(ae, x_train)
        view_reference = view_reconstruction_score(x_train)
        latent_scaler = StandardScaler().fit(latent_train)
        latent_model = LedoitWolf().fit(latent_scaler.transform(latent_train))
        latent_reference = latent_model.mahalanobis(latent_scaler.transform(latent_train))
        alpha = float(params.get("energy_alpha", 0.65))

        def raw(x):
            _, _, latent, _ = ae_outputs(ae, x)
            view_score = view_reconstruction_score(x)
            latent_score = latent_model.mahalanobis(latent_scaler.transform(latent))
            return alpha * empirical_percentile(view_reference, view_score) + (1.0 - alpha) * empirical_percentile(
                latent_reference, latent_score
            )
    elif kind == "deep_svdd":
        model = fit_svdd(x_train, params["hidden"], params["embed_dim"], params["epochs"], seed)
        raw = lambda x: svdd_scores(model, x)
    elif kind == "isolation_forest":
        scaler = StandardScaler().fit(x_train)
        model = IsolationForest(n_estimators=400, random_state=seed, contamination="auto").fit(scaler.transform(x_train))
        raw = lambda x: -model.score_samples(scaler.transform(x))
    else:
        raise ValueError(kind)
    reference = np.asarray(raw(x_train), dtype=float)
    return lambda x: continuous_standard_score(reference, raw(np.asarray(x, dtype=float)))


def fit_candidate(config, train_profile, train_compact, train_change, train_station, seed):
    params = config["params"]
    station_transformer = None
    if config.get("station_conditioned", False):
        station_transformer = StationShrinkTransformer(params["station_kappa"]).fit(train_profile, train_station)
        state_train = station_transformer.transform(train_profile, train_station)
    else:
        state_train = np.asarray(train_profile, dtype=float)
    state_head = fit_state_head(config["state_head"], state_train, params, seed)

    transition_head = None
    if config.get("transition_head"):
        transition_head = fit_transition_head(
            config["transition_head"], train_compact[np.asarray(train_change, dtype=bool)], params, seed + 101
        )
    transition_weight = float(params.get("transition_weight", 0.0))

    def score(profile, compact, is_change, stations):
        profile = np.asarray(profile, dtype=float)
        compact = np.asarray(compact, dtype=float)
        is_change = np.asarray(is_change, dtype=bool)
        state_x = (
            station_transformer.transform(profile, stations) if station_transformer is not None else profile
        )
        state_evidence = state_head(state_x)
        if transition_head is None:
            return state_evidence
        transition_evidence = np.zeros(len(profile), dtype=float)
        if is_change.any():
            transition_evidence[is_change] = transition_head(compact[is_change])
        return state_evidence + transition_weight * transition_evidence

    return score


def candidate_registry():
    configs = []

    def add(family, state_head, transition_head=None, station_conditioned=False, **params):
        configs.append(
            {
                "config_id": f"masked_transition_{len(configs):02d}",
                "family": family,
                "state_head": state_head,
                "transition_head": transition_head,
                "station_conditioned": bool(station_conditioned),
                "params": params,
            }
        )

    add("Baseline Raw OCSVM", "ocsvm", nu=0.10, gamma=1.0)

    for transition_head in ["mahalanobis", "ocsvm"]:
        for weight in [0.15, 0.30]:
            add(
                "Profile-Anchor Transition Boost",
                "ocsvm",
                transition_head=transition_head,
                nu=0.10,
                gamma=1.0,
                transition_nu=0.10,
                transition_weight=weight,
            )

    for hidden, latent in [([32, 16], 3), ([64, 32], 4)]:
        for mask_rate in [0.15, 0.30]:
            for gamma in ["scale", 1.0]:
                add(
                    "Masked AE+OCSVM",
                    "masked_ae_ocsvm",
                    hidden=hidden,
                    latent=latent,
                    mask_rate=mask_rate,
                    gamma=gamma,
                    nu=0.05,
                    epochs=180,
                )

    for kappa in [5.0, 20.0]:
        for mask_rate in [0.15, 0.30]:
            for hidden, latent in [([32, 16], 3), ([64, 32], 4)]:
                add(
                    "Station-Conditioned Masked AE+OCSVM",
                    "masked_ae_ocsvm",
                    station_conditioned=True,
                    station_kappa=kappa,
                    hidden=hidden,
                    latent=latent,
                    mask_rate=mask_rate,
                    gamma="scale",
                    nu=0.05,
                    epochs=180,
                )

    for transition_head in ["mahalanobis", "svdd", "masked_ae"]:
        for mask_rate in [0.15, 0.30]:
            for weight in [0.15, 0.30]:
                add(
                    "Dual-Branch Masked-AE Transition Boost",
                    "masked_ae_ocsvm",
                    transition_head=transition_head,
                    hidden=[64, 32],
                    latent=4,
                    mask_rate=mask_rate,
                    gamma="scale",
                    nu=0.05,
                    epochs=180,
                    transition_epochs=160,
                    transition_weight=weight,
                )

    for hidden, latent in [([32, 16], 3), ([64, 32], 4)]:
        for mask_rate in [0.15, 0.30]:
            for alpha in [0.50, 0.75]:
                add(
                    "Multi-View Masked-AE Energy",
                    "multiview_masked_ae",
                    hidden=hidden,
                    latent=latent,
                    mask_rate=mask_rate,
                    energy_alpha=alpha,
                    epochs=180,
                )

    for hidden, embed_dim in [([32, 16], 3), ([64, 32], 5)]:
        for transition_head in [None, "svdd"]:
            for weight in ([0.0] if transition_head is None else [0.15, 0.30]):
                add(
                    "Neural Deep-SVDD Dual Score",
                    "deep_svdd",
                    transition_head=transition_head,
                    hidden=hidden,
                    embed_dim=embed_dim,
                    epochs=220,
                    transition_epochs=180,
                    transition_weight=weight,
                )

    for hidden, latent in [([32, 16], 3), ([64, 32], 4)]:
        for mask_rate in [0.15, 0.30]:
            for alpha in [0.35, 0.65]:
                add(
                    "Masked-AE Reconstruction-Latent Energy",
                    "masked_ae_energy",
                    hidden=hidden,
                    latent=latent,
                    mask_rate=mask_rate,
                    energy_alpha=alpha,
                    epochs=180,
                )
    return configs


def constrained_threshold(scores, event_types):
    scores = np.asarray(scores, dtype=float)
    event_types = np.asarray(event_types, dtype=str)
    update_mask = event_types == "user_update"
    candidates = np.r_[np.nextafter(np.unique(scores), np.inf), np.inf]
    for threshold in candidates:
        predictions = scores >= threshold
        overall_tnr = float(np.mean(~predictions))
        update_tnr = float(np.mean(~predictions[update_mask])) if update_mask.any() else 1.0
        if overall_tnr + 1e-12 >= OVERALL_TNR_TARGET and update_tnr + 1e-12 >= UPDATE_TNR_TARGET:
            return float(threshold)
    raise RuntimeError("No threshold satisfies both normal-subgroup constraints")


def metric_row(y_true, scores, threshold):
    y_true = np.asarray(y_true, dtype=int)
    prediction = (np.asarray(scores, dtype=float) >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, prediction, average="binary", zero_division=0
    )
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "fpr": float(fp / max(fp + tn, 1)), "tnr": float(tn / max(fp + tn, 1)),
    }


def wilson_interval(successes, total, confidence=0.95):
    if total <= 0:
        return np.nan, np.nan
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def evaluate_fold(data: DataBundle, config, fold, cache_dir: Path, registry_hash: str):
    cache_path = cache_dir / f"{config['config_id']}_fold{fold['fold']}.joblib"
    if cache_path.exists():
        cached = joblib.load(cache_path)
        if cached.get("registry_hash") == registry_hash:
            return cached

    train_mask = source_mask(data.source_ids, fold["train_source_ids"])
    val_mask = source_mask(data.source_ids, fold["validation_source_ids"])
    seed = SEED + 1000 * int(fold["fold"]) + int(config["config_id"].split("_")[-1])
    scorer = fit_candidate(
        config,
        data.profile[train_mask], data.compact[train_mask], data.is_change[train_mask], data.station_ids[train_mask], seed,
    )
    normal_scores = scorer(
        data.profile[val_mask], data.compact[val_mask], data.is_change[val_mask], data.station_ids[val_mask]
    )
    event_types = np.asarray(data.pool["state_event_types"], dtype=str)[val_mask]
    threshold = constrained_threshold(normal_scores, event_types)
    normal_prediction = normal_scores >= threshold
    update_mask = event_types == "user_update"

    per_attack = []
    all_attack_scores = []
    for attack_type in ATTACK_TYPES:
        eligible = np.asarray(data.pool["attack_state_indices"][attack_type], dtype=int)
        selected = eligible[source_mask(data.states.iloc[eligible].source_id.to_numpy(), fold["validation_source_ids"])]
        attack_scores = scorer(
            np.asarray(data.pool["attack_profile_features"][attack_type])[selected],
            data.attack_compact[attack_type][selected],
            np.ones(len(selected), dtype=bool),
            data.station_ids[selected],
        )
        all_attack_scores.append(attack_scores)
        metrics = metric_row(
            np.r_[np.zeros(len(normal_scores), dtype=int), np.ones(len(attack_scores), dtype=int)],
            np.r_[normal_scores, attack_scores], threshold,
        )
        per_attack.append({"attack_type": attack_type, "attack_count": len(attack_scores), **metrics})

    result = {
        "registry_hash": registry_hash,
        "config": config,
        "fold": int(fold["fold"]),
        "threshold": threshold,
        "normal_count": len(normal_scores),
        "normal_tnr": float(np.mean(~normal_prediction)),
        "activation_tnr": float(np.mean(~normal_prediction[~update_mask])),
        "update_count": int(update_mask.sum()),
        "update_tnr": float(np.mean(~normal_prediction[update_mask])) if update_mask.any() else np.nan,
        "macro_f1": float(np.mean([row["f1"] for row in per_attack])),
        "aggregate": metric_row(
            np.r_[np.zeros(len(normal_scores), dtype=int), np.ones(sum(map(len, all_attack_scores)), dtype=int)],
            np.concatenate([normal_scores] + all_attack_scores), threshold,
        ),
        "per_attack": per_attack,
    }
    joblib.dump(result, cache_path)
    return result


def final_evaluation(data: DataBundle, selected_configs, results_dir: Path):
    train_ids = data.plan["final_train_source_ids"]
    cal_ids = data.plan["calibration_source_ids"]
    test_ids = data.plan["test_source_ids"]
    train_mask = source_mask(data.source_ids, train_ids)
    cal_mask = source_mask(data.source_ids, cal_ids)
    test_mask = source_mask(data.source_ids, test_ids)
    event_types_all = np.asarray(data.pool["state_event_types"], dtype=str)
    final_rows, per_attack_rows, update_rows, payload = [], [], [], {}

    for model_index, config in enumerate(selected_configs):
        scorer = fit_candidate(
            config,
            data.profile[train_mask], data.compact[train_mask], data.is_change[train_mask], data.station_ids[train_mask],
            SEED + 20000 + model_index,
        )
        cal_scores = scorer(
            data.profile[cal_mask], data.compact[cal_mask], data.is_change[cal_mask], data.station_ids[cal_mask]
        )
        threshold = constrained_threshold(cal_scores, event_types_all[cal_mask])
        test_scores = scorer(
            data.profile[test_mask], data.compact[test_mask], data.is_change[test_mask], data.station_ids[test_mask]
        )
        test_event_types = event_types_all[test_mask]
        test_prediction = test_scores >= threshold
        update_mask = test_event_types == "user_update"
        update_fp = int(test_prediction[update_mask].sum())
        update_ci = wilson_interval(update_fp, int(update_mask.sum()))

        all_attack_scores, all_attack_state_ids = [], []
        for attack_type in ATTACK_TYPES:
            eligible = np.asarray(data.pool["attack_state_indices"][attack_type], dtype=int)
            selected = eligible[source_mask(data.states.iloc[eligible].source_id.to_numpy(), test_ids)]
            attack_scores = scorer(
                np.asarray(data.pool["attack_profile_features"][attack_type])[selected],
                data.attack_compact[attack_type][selected],
                np.ones(len(selected), dtype=bool),
                data.station_ids[selected],
            )
            all_attack_scores.append(attack_scores)
            all_attack_state_ids.append(selected)
            metrics = metric_row(
                np.r_[np.zeros(len(test_scores), dtype=int), np.ones(len(attack_scores), dtype=int)],
                np.r_[test_scores, attack_scores], threshold,
            )
            per_attack_rows.append(
                {"config_id": config["config_id"], "family": config["family"], "attack_type": attack_type,
                 "attack_count": len(attack_scores), **metrics}
            )

        concatenated_attacks = np.concatenate(all_attack_scores)
        aggregate = metric_row(
            np.r_[np.zeros(len(test_scores), dtype=int), np.ones(len(concatenated_attacks), dtype=int)],
            np.r_[test_scores, concatenated_attacks], threshold,
        )
        final_rows.append(
            {
                "config_id": config["config_id"], "family": config["family"],
                "selected_configuration": json.dumps(config, sort_keys=True),
                "calibration_threshold": threshold,
                "calibration_normal_tnr": float(np.mean(cal_scores < threshold)),
                "calibration_update_tnr": float(np.mean(cal_scores[event_types_all[cal_mask] == "user_update"] < threshold)),
                "test_normal_count": len(test_scores), "test_update_count": int(update_mask.sum()),
                "test_update_tnr": float(np.mean(~test_prediction[update_mask])),
                "test_update_fp": update_fp, "test_update_fpr_ci_low": update_ci[0],
                "test_update_fpr_ci_high": update_ci[1], **aggregate,
            }
        )

        test_state_ids = np.flatnonzero(test_mask)
        for state_id, score, prediction in zip(test_state_ids[update_mask], test_scores[update_mask], test_prediction[update_mask]):
            row = data.states.iloc[state_id]
            update_rows.append(
                {"config_id": config["config_id"], "family": config["family"], "state_id": int(state_id),
                 "source_id": int(row.source_id), "source_uid": row.source_uid, "edit_kind": row.edit_kind,
                 "score": float(score), "threshold": float(threshold), "false_alarm": int(prediction)}
            )
        payload[config["config_id"]] = {
            "config": config, "threshold": threshold, "normal_scores": test_scores,
            "attack_scores": concatenated_attacks, "attack_state_ids": np.concatenate(all_attack_state_ids),
        }

    final_df = pd.DataFrame(final_rows)
    per_attack_df = pd.DataFrame(per_attack_rows)
    update_df = pd.DataFrame(update_rows)
    final_df.to_csv(results_dir / "masked_transition_selected_model_final_test_summary.csv", index=False)
    per_attack_df.to_csv(results_dir / "masked_transition_selected_model_per_attack_results.csv", index=False)
    update_df.to_csv(results_dir / "masked_transition_observed_user_update_predictions.csv", index=False)
    joblib.dump(payload, results_dir / "masked_transition_final_test_score_payload.joblib")
    return final_df, per_attack_df, update_df


def make_figures(cv_summary, selected_cv, final_df, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    merged = selected_cv[["family", "config_id", "robust_cv_score"]].merge(
        final_df[["family", "config_id", "f1", "fpr", "test_update_tnr"]], on=["family", "config_id"], how="left"
    ).sort_values("f1")
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10})
    fig, ax = plt.subplots(figsize=(8.7, 5.6))
    colors = ["#4472C4" if family != "Baseline Raw OCSVM" else "#7F7F7F" for family in merged.family]
    ax.barh(merged.family, merged.f1, color=colors)
    ax.set_xlabel("Final-test F1 (configuration selected by CV)")
    ax.set_xlim(0, 1)
    for y, value in enumerate(merged.f1):
        ax.text(min(value + 0.01, 0.96), y, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(figures_dir / "masked_transition_model_f1_summary_notitle.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "masked_transition_model_f1_summary_notitle.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for _, row in merged.iterrows():
        ax.scatter(row.fpr, row.f1, s=70, label=row.family)
    ax.set_xlabel("Final-test FPR")
    ax.set_ylabel("Final-test F1")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "masked_transition_model_f1_vs_fpr_notitle.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "masked_transition_model_f1_vs_fpr_notitle.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment(root: Path, smoke: bool = False):
    configure_seed(SEED)
    root = Path(root).resolve()
    results_dir = root / "results" / "session_level" / "masked_transition_models"
    figures_dir = root / "figures" / "session_level" / "masked_transition_models"
    cache_dir = results_dir / "task_cache"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(root)
    configs = candidate_registry()
    registry_payload = {
        "created_before_cv": True,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": SEED,
        "folds": N_FOLDS,
        "selection_metric": "mean_macro_f1_minus_0.5_fold_std",
        "normal_tnr_constraint": OVERALL_TNR_TARGET,
        "observed_update_tnr_constraint": UPDATE_TNR_TARGET,
        "configs": configs,
    }
    registry_hash = hashlib.sha256(json.dumps(registry_payload, sort_keys=True).encode("utf-8")).hexdigest()
    registry_payload["registry_sha256"] = registry_hash
    (results_dir / "masked_transition_candidate_registry.json").write_text(
        json.dumps(json_ready(registry_payload), indent=2), encoding="utf-8"
    )

    folds = data.plan["folds"]
    run_configs = configs
    run_folds = folds
    if smoke:
        smoke_families = sorted({config["family"] for config in configs})
        run_configs = [next(config for config in configs if config["family"] == family) for family in smoke_families]
        run_folds = folds[:1]

    tasks = [(config, fold) for config in run_configs for fold in run_folds]
    started = time.time()
    results = Parallel(n_jobs=min(MAX_WORKERS, len(tasks)), backend="loky", verbose=10)(
        delayed(evaluate_fold)(data, config, fold, cache_dir, registry_hash) for config, fold in tasks
    )
    if smoke:
        smoke_payload = {
            "status": "PASS", "tasks": len(results), "families": sorted({r["config"]["family"] for r in results}),
            "runtime_sec": time.time() - started,
        }
        (results_dir / "masked_transition_smoke_test.json").write_text(
            json.dumps(json_ready(smoke_payload), indent=2), encoding="utf-8"
        )
        return {"smoke": smoke_payload}

    cv_rows, per_attack_rows = [], []
    for result in results:
        config = result["config"]
        cv_rows.append(
            {
                "config_id": config["config_id"], "family": config["family"], "fold": result["fold"],
                "macro_f1": result["macro_f1"], "aggregate_f1": result["aggregate"]["f1"],
                "normal_tnr": result["normal_tnr"], "activation_tnr": result["activation_tnr"],
                "update_count": result["update_count"], "update_tnr": result["update_tnr"],
                "threshold": result["threshold"],
            }
        )
        for row in result["per_attack"]:
            per_attack_rows.append(
                {"config_id": config["config_id"], "family": config["family"], "fold": result["fold"], **row}
            )
    cv_df = pd.DataFrame(cv_rows)
    cv_per_attack_df = pd.DataFrame(per_attack_rows)
    cv_summary = cv_df.groupby(["config_id", "family"], as_index=False).agg(
        cv_macro_f1_mean=("macro_f1", "mean"), cv_macro_f1_std=("macro_f1", "std"),
        cv_aggregate_f1_mean=("aggregate_f1", "mean"), cv_normal_tnr_min=("normal_tnr", "min"),
        cv_update_tnr_mean=("update_tnr", "mean"), cv_update_tnr_min=("update_tnr", "min"),
        cv_update_count=("update_count", "sum"),
    )
    cv_summary["robust_cv_score"] = cv_summary.cv_macro_f1_mean - STABILITY_PENALTY * cv_summary.cv_macro_f1_std
    cv_summary = cv_summary.sort_values(
        ["robust_cv_score", "cv_update_tnr_mean", "cv_normal_tnr_min"], ascending=[False, False, False]
    ).reset_index(drop=True)
    cv_df.to_csv(results_dir / "masked_transition_cv_fold_results.csv", index=False)
    cv_per_attack_df.to_csv(results_dir / "masked_transition_cv_per_attack_results.csv", index=False)
    cv_summary.to_csv(results_dir / "masked_transition_cv_configuration_summary.csv", index=False)

    selected_cv = (
        cv_summary.sort_values(
            ["family", "robust_cv_score", "cv_update_tnr_mean", "cv_normal_tnr_min"],
            ascending=[True, False, False, False],
        ).groupby("family", as_index=False).first()
    )
    config_by_id = {config["config_id"]: config for config in configs}
    selected_configs = [config_by_id[config_id] for config_id in selected_cv.config_id]
    selected_cv.to_csv(results_dir / "masked_transition_family_selected_configurations.csv", index=False)

    # Final test scoring begins only after all family and global selections are frozen above.
    final_df, per_attack_df, update_df = final_evaluation(data, selected_configs, results_dir)
    combined = selected_cv.merge(final_df, on=["config_id", "family"], how="left")
    combined.to_csv(results_dir / "masked_transition_model_comparison_summary.csv", index=False)
    formal = selected_cv.sort_values(
        ["robust_cv_score", "cv_update_tnr_mean", "cv_normal_tnr_min"], ascending=[False, False, False]
    ).iloc[0]
    formal_final = final_df[final_df.config_id == formal.config_id].iloc[0]

    design = {
        "status": "PASS",
        "dataset": "caltech_sessions_2021.json",
        "candidate_registry_sha256": registry_hash,
        "candidate_grid_frozen_before_cv": True,
        "model_families": sorted({config["family"] for config in configs}),
        "candidate_count": len(configs), "cv_tasks": len(tasks), "maximum_workers": MAX_WORKERS,
        "source_grouped_folds": N_FOLDS,
        "fixed_attack_pool_reused": True,
        "final_test_used_for_selection": False,
        "normal_tnr_constraint": OVERALL_TNR_TARGET,
        "observed_update_tnr_constraint": UPDATE_TNR_TARGET,
        "formal_selected_config": config_by_id[formal.config_id],
        "formal_selected_cv": formal.to_dict(),
        "formal_final_test_metrics": formal_final.to_dict(),
        "runtime_sec": time.time() - started,
    }
    (results_dir / "masked_transition_experiment_design.json").write_text(
        json.dumps(json_ready(design), indent=2), encoding="utf-8"
    )
    make_figures(cv_summary, selected_cv, final_df, figures_dir)

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    return {
        "design": design, "cv_summary": cv_summary, "selected_cv": selected_cv,
        "final_summary": combined, "per_attack": per_attack_df, "updates": update_df,
    }
