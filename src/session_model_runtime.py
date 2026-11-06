"""Unified runtime for every CV-selected session-level detector family."""

from __future__ import annotations

import json
import hashlib
import math
import random
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from masked_transition_model_search_core import compact_transition, fit_candidate as fit_masked_candidate


def configure_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)


class FlexibleAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        encoder = []
        previous = input_dim
        for width in hidden_dims:
            encoder += [nn.Linear(previous, width), nn.ReLU()]
            previous = width
        encoder.append(nn.Linear(previous, latent_dim))
        self.encoder = nn.Sequential(*encoder)
        decoder = []
        previous = latent_dim
        for width in reversed(hidden_dims):
            decoder += [nn.Linear(previous, width), nn.ReLU()]
            previous = width
        decoder.append(nn.Linear(previous, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent), latent


class AffineCoupling(nn.Module):
    def __init__(self, dimension, mask, hidden):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dimension, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * dimension),
        )

    def forward(self, x):
        masked = x * self.mask
        scale, shift = self.net(masked).chunk(2, dim=1)
        scale = torch.tanh(scale) * (1 - self.mask)
        shift = shift * (1 - self.mask)
        output = masked + (1 - self.mask) * (x * torch.exp(scale) + shift)
        return output, scale.sum(dim=1)


class RealNVP(nn.Module):
    def __init__(self, dimension, layers=6, hidden=64):
        super().__init__()
        self.blocks = nn.ModuleList([
            AffineCoupling(
                dimension,
                ((torch.arange(dimension) + index) % 2).float().unsqueeze(0),
                hidden,
            )
            for index in range(layers)
        ])

    def log_prob(self, x):
        latent = x
        log_det = torch.zeros(len(x), device=x.device)
        for block in self.blocks:
            latent, term = block(latent)
            log_det += term
        return -0.5 * (latent ** 2 + math.log(2 * math.pi)).sum(dim=1) + log_det


class ScoreMLP(nn.Module):
    def __init__(self, input_dim, hidden=(64, 32), output_dim=1):
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden:
            layers += [nn.Linear(previous, width), nn.ReLU()]
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_ae(x_train, hidden, latent, epochs, learning_rate, seed):
    configure_seed(seed)
    scaler = StandardScaler().fit(x_train)
    x = scaler.transform(x_train).astype(np.float32)
    model = FlexibleAE(x.shape[1], hidden, latent)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    generator = torch.Generator().manual_seed(int(seed) + 17)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x)),
        batch_size=min(64, len(x)), shuffle=True, generator=generator,
    )
    model.train()
    for _ in range(int(epochs)):
        for (batch,) in loader:
            reconstruction, _ = model(batch)
            loss = ((reconstruction - batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return {"scaler": scaler, "model": model}


def ae_features(fit, x):
    scaled = fit["scaler"].transform(x).astype(np.float32)
    with torch.no_grad():
        reconstruction, latent = fit["model"](torch.from_numpy(scaled))
    return np.column_stack([scaled - reconstruction.numpy(), latent.numpy()])


def train_flow(x_train, seed, layers=6, hidden=64, epochs=120):
    configure_seed(seed)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x = ((x_train - mean) / std).astype(np.float32)
    flow = RealNVP(x.shape[1], layers=layers, hidden=hidden)
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(int(seed) + 23)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x)),
        batch_size=min(128, len(x)), shuffle=True, generator=generator,
    )
    flow.train()
    for _ in range(int(epochs)):
        for (batch,) in loader:
            loss = -flow.log_prob(batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    flow.eval()
    return {"mean": mean, "std": std, "flow": flow}


def flow_score(fit, x):
    standardized = ((x - fit["mean"]) / fit["std"]).astype(np.float32)
    with torch.no_grad():
        return (-fit["flow"].log_prob(torch.from_numpy(standardized))).numpy()


def train_deep_svdd(x_train, seed, hidden=(64, 32), embed_dim=4, epochs=220):
    configure_seed(seed)
    scaler = StandardScaler().fit(x_train)
    x = scaler.transform(x_train).astype(np.float32)
    tensor = torch.from_numpy(x)
    net = ScoreMLP(x.shape[1], hidden=hidden, output_dim=embed_dim)
    with torch.no_grad():
        center = net.net(tensor).mean(dim=0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-6)
    generator = torch.Generator().manual_seed(int(seed) + 29)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tensor), batch_size=min(64, len(x)),
        shuffle=True, generator=generator,
    )
    for _ in range(int(epochs)):
        for (batch,) in loader:
            embedding = net.net(batch)
            loss = ((embedding - center) ** 2).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    net.eval()
    return {"scaler": scaler, "net": net, "center": center}


def make_tail_pseudo(x, seed, multiplier=2):
    rng = np.random.default_rng(seed)
    pseudo = x[rng.integers(0, len(x), size=len(x) * multiplier)].copy()
    dimensions = rng.integers(1, max(2, x.shape[1] // 2 + 1), size=len(pseudo))
    for index, count in enumerate(dimensions):
        columns = rng.choice(x.shape[1], size=count, replace=False)
        pseudo[index, columns] += (
            rng.choice([-1.0, 1.0], size=len(columns))
            * rng.uniform(2.5, 4.5, size=len(columns))
        )
    return pseudo


def train_devnet(x_train, seed, hidden=(64, 32), epochs=220):
    configure_seed(seed)
    scaler = StandardScaler().fit(x_train)
    normal = scaler.transform(x_train).astype(np.float32)
    pseudo = make_tail_pseudo(normal, seed + 101)
    x = np.vstack([normal, pseudo]).astype(np.float32)
    y = np.r_[np.zeros(len(normal)), np.ones(len(pseudo))].astype(np.float32)
    order = np.random.default_rng(seed).permutation(len(x))
    model = ScoreMLP(x.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(int(seed) + 31)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x[order]), torch.from_numpy(y[order])),
        batch_size=64, shuffle=True, generator=generator,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(int(epochs)):
        for batch, labels in loader:
            loss = loss_fn(model(batch), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return {"scaler": scaler, "model": model}


def select_matrix(protocol, profile, transition):
    if protocol == "profile_only":
        return np.asarray(profile, dtype=float)
    if protocol == "transition_aware":
        return np.asarray(transition, dtype=float)
    raise ValueError(protocol)


def fit_basic_scorer(algorithm, x_train, params, seed):
    scaler = StandardScaler().fit(x_train)
    x = scaler.transform(x_train)
    if algorithm == "ocsvm":
        model = OneClassSVM(kernel="rbf", nu=params["nu"], gamma=params["gamma"]).fit(x)
        return lambda values: -model.decision_function(scaler.transform(values)).reshape(-1)
    if algorithm == "knn":
        model = NearestNeighbors(n_neighbors=min(params["k"], len(x))).fit(x)
        return lambda values: model.kneighbors(scaler.transform(values))[0].mean(axis=1)
    if algorithm == "mahalanobis":
        model = LedoitWolf().fit(x)
        return lambda values: model.mahalanobis(scaler.transform(values))
    if algorithm == "gmm":
        model = GaussianMixture(
            n_components=params["components"], covariance_type="full",
            reg_covar=1e-5, random_state=seed,
        ).fit(x)
        return lambda values: -model.score_samples(scaler.transform(values))
    if algorithm == "iforest":
        model = IsolationForest(n_estimators=400, max_samples="auto", random_state=seed).fit(x)
        return lambda values: -model.score_samples(scaler.transform(values))
    if algorithm == "lof":
        model = LocalOutlierFactor(
            n_neighbors=min(params["k"], max(2, len(x) - 1)), novelty=True,
        ).fit(x)
        return lambda values: -model.score_samples(scaler.transform(values))
    raise ValueError(algorithm)


def fit_original_scorer(config, train_profile, train_transition, train_is_change, seed):
    protocol = config["protocol"]
    algorithm = config["algorithm"]
    params = config["params"]
    if protocol == "context_stratified":
        change_mask = np.asarray(train_is_change, dtype=bool)
        if change_mask.sum() < 8:
            raise RuntimeError("Insufficient observed updates for context-stratified fitting")
        activation_raw = fit_basic_scorer(algorithm, train_profile[~change_mask], params, seed)
        change_raw = fit_basic_scorer(algorithm, train_transition[change_mask, :10], params, seed + 1)
        activation_reference = activation_raw(train_profile[~change_mask])
        change_reference = change_raw(train_transition[change_mask, :10])
        activation_mu, activation_sd = activation_reference.mean(), activation_reference.std() + 1e-8
        change_mu, change_sd = change_reference.mean(), change_reference.std() + 1e-8

        def score(profile, transition, is_change):
            mask = np.asarray(is_change, dtype=bool)
            output = np.empty(len(profile), dtype=float)
            if (~mask).any():
                output[~mask] = (activation_raw(profile[~mask]) - activation_mu) / activation_sd
            if mask.any():
                output[mask] = (change_raw(transition[mask, :10]) - change_mu) / change_sd
            return output

        return score

    train_x = select_matrix(protocol, train_profile, train_transition)
    if algorithm in {"ocsvm", "knn", "mahalanobis", "gmm", "iforest", "lof"}:
        head = fit_basic_scorer(algorithm, train_x, params, seed)
        return lambda profile, transition, is_change: head(select_matrix(protocol, profile, transition))
    if algorithm.startswith("ae_") or algorithm == "ae_nf":
        autoencoder = train_ae(
            train_x, params["hidden"], params["latent"], params.get("epochs", 260),
            params.get("lr", 1e-3), seed,
        )
        representation = ae_features(autoencoder, train_x)
        rep_scaler = StandardScaler().fit(representation)
        scaled_representation = rep_scaler.transform(representation)
        if algorithm == "ae_ocsvm":
            model = OneClassSVM(kernel="rbf", nu=params["nu"], gamma=params["gamma"]).fit(scaled_representation)
            rep_score = lambda values: -model.decision_function(rep_scaler.transform(values)).reshape(-1)
        elif algorithm == "ae_mahalanobis":
            model = LedoitWolf().fit(scaled_representation)
            rep_score = lambda values: model.mahalanobis(rep_scaler.transform(values))
        elif algorithm == "ae_gmm":
            model = GaussianMixture(
                n_components=params["components"], covariance_type="full",
                reg_covar=1e-5, random_state=seed,
            ).fit(scaled_representation)
            rep_score = lambda values: -model.score_samples(rep_scaler.transform(values))
        else:
            flow = train_flow(
                representation, seed + 11, layers=params.get("flow_layers", 6),
                hidden=params.get("flow_hidden", 64), epochs=params.get("flow_epochs", 120),
            )
            rep_score = lambda values: flow_score(flow, values)

        def score(profile, transition, is_change):
            values = select_matrix(protocol, profile, transition)
            return rep_score(ae_features(autoencoder, values))

        return score
    if algorithm == "deep_svdd":
        fit = train_deep_svdd(train_x, seed, hidden=params["hidden"], embed_dim=params["embed_dim"])

        def score(profile, transition, is_change):
            values = fit["scaler"].transform(select_matrix(protocol, profile, transition)).astype(np.float32)
            with torch.no_grad():
                embedding = fit["net"].net(torch.from_numpy(values))
            return ((embedding - fit["center"]) ** 2).sum(dim=1).numpy()

        return score
    if algorithm == "devnet":
        fit = train_devnet(train_x, seed, hidden=params["hidden"])

        def score(profile, transition, is_change):
            values = fit["scaler"].transform(select_matrix(protocol, profile, transition)).astype(np.float32)
            with torch.no_grad():
                return fit["model"](torch.from_numpy(values)).numpy()

        return score
    raise ValueError(config)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_unified_registry(root: Path):
    root = Path(root).resolve()
    original_configs = json.loads((root / "results/session_level/candidate_configurations.json").read_text(encoding="utf-8"))
    original_by_id = {item["config_id"]: item for item in original_configs}
    original_selected = pd.read_csv(
        root / "results/session_level/boundary_density_family_selected_final_test_summary.csv"
    )

    masked_dir = root / "results" / "session_level" / "masked_transition_models"
    masked_registry = json.loads(
        (masked_dir / "masked_transition_candidate_registry.json").read_text(encoding="utf-8")
    )
    masked_selected = pd.read_csv(
        masked_dir / "masked_transition_family_selected_configurations.csv"
    )
    masked_configs = []
    for item in masked_registry["configs"]:
        if item["config_id"] == "masked_transition_00":
            continue
        clean = json.loads(json.dumps(item))
        masked_configs.append(clean)

    selected = []
    for row in original_selected.itertuples():
        config = json.loads(json.dumps(original_by_id[row.config_id]))
        selected.append({
            "model_id": config["config_id"],
            "display_name": f"{config['family']} [{config['protocol'].replace('_', ' ')}]",
            "selection_group": f"{config['family']}::{config['protocol']}",
            "model_block": "standard",
            "config": config,
            "robust_cv_score": float(row.robust_cv_score),
        })
    masked_by_id = {item["config_id"]: item for item in masked_configs}
    for row in masked_selected.itertuples():
        if row.config_id == "masked_transition_00":
            continue
        config = masked_by_id[row.config_id]
        selected.append({
            "model_id": row.config_id,
            "display_name": config["family"],
            "selection_group": config["family"],
            "model_block": "masked_transition",
            "config": config,
            "robust_cv_score": float(row.robust_cv_score),
        })

    unified_configs = [
        {"model_block": "standard", **json.loads(json.dumps(item))}
        for item in original_configs
    ] + [
        {"model_block": "masked_transition", **json.loads(json.dumps(item))}
        for item in masked_configs
    ]
    if len(unified_configs) != 89 or len(selected) != 34:
        raise RuntimeError(f"Unexpected unified registry size: {len(unified_configs)} configs, {len(selected)} selected")
    return {"configs": unified_configs, "selected": selected}


def export_unified_session_artifacts(root: Path, registry=None):
    root = Path(root).resolve()
    registry = registry or load_unified_registry(root)
    session_dir = root / "results" / "session_level"
    original = pd.read_csv(session_dir / "boundary_density_family_selected_final_test_summary.csv")
    masked_dir = session_dir / "masked_transition_models"
    masked = pd.read_csv(masked_dir / "masked_transition_model_comparison_summary.csv")
    rows = []
    for record in registry["selected"]:
        config = record["config"]
        if record["model_block"] == "standard":
            source = original[original["config_id"].eq(config["config_id"])].iloc[0].to_dict()
            protocol = str(source["protocol"])
        else:
            source = masked[masked["config_id"].eq(config["config_id"])].iloc[0].to_dict()
            if config.get("station_conditioned"):
                protocol = "context_stratified"
            elif config.get("transition_head"):
                protocol = "transition_aware"
            else:
                protocol = "profile_only"
        rows.append({
            "model_id": record["model_id"],
            "family": config["family"],
            "protocol": protocol,
            "model_block": record["model_block"],
            "selected_configuration": json.dumps(config, sort_keys=True),
            "cv_macro_f1_mean": float(source["cv_macro_f1_mean"]),
            "cv_macro_f1_std": float(source["cv_macro_f1_std"]),
            "robust_cv_score": float(source["robust_cv_score"]),
            "calibration_normal_tnr": float(source["calibration_normal_tnr"]),
            "calibration_update_tnr": float(source["calibration_update_tnr"]),
            "test_normal_count": int(source["test_normal_count"]),
            "test_update_count": int(source["test_update_count"]),
            "test_update_tnr": float(source["test_update_tnr"]),
            "tn": int(source["tn"]), "fp": int(source["fp"]),
            "fn": int(source["fn"]), "tp": int(source["tp"]),
            "precision": float(source["precision"]), "recall": float(source["recall"]),
            "f1": float(source["f1"]), "fpr": float(source["fpr"]), "tnr": float(source["tnr"]),
        })
    selected = pd.DataFrame(rows).sort_values("robust_cv_score", ascending=False).reset_index(drop=True)
    selected["selected_globally"] = selected.index == 0
    selected.to_csv(session_dir / "unified_selected_model_summary.csv", index=False)
    selected.iloc[[0]].to_csv(
        session_dir / "unified_global_selected_model_summary.csv", index=False,
    )
    family_indices = selected.groupby("family")["robust_cv_score"].idxmax()
    family = selected.loc[family_indices].sort_values("robust_cv_score", ascending=False).reset_index(drop=True)
    family.to_csv(session_dir / "unified_family_best_summary.csv", index=False)

    global_row = selected.iloc[0]
    global_record = next(
        record for record in registry["selected"]
        if record["model_id"] == global_row["model_id"]
    )
    global_model_id = str(global_row["model_id"])
    if global_record["model_block"] == "standard":
        per_attack = pd.read_csv(
            session_dir / "boundary_density_family_selected_per_attack_results.csv"
        )
        update_predictions = pd.read_csv(
            session_dir / "boundary_density_family_user_update_predictions.csv"
        )
        score_payload = joblib.load(session_dir / "final_test_score_payload.joblib")[global_model_id]
    else:
        per_attack = pd.read_csv(
            masked_dir / "masked_transition_selected_model_per_attack_results.csv"
        )
        update_predictions = pd.read_csv(
            masked_dir / "masked_transition_observed_user_update_predictions.csv"
        )
        score_payload = joblib.load(
            masked_dir / "masked_transition_final_test_score_payload.joblib"
        )[global_model_id]
    per_attack = per_attack[per_attack["config_id"].eq(global_model_id)].copy()
    update_predictions = update_predictions[
        update_predictions["config_id"].eq(global_model_id)
    ].copy()
    for frame in (per_attack, update_predictions):
        if "protocol" not in frame:
            frame.insert(2, "protocol", str(global_row["protocol"]))
    per_attack.to_csv(
        session_dir / "unified_global_selected_model_per_attack_results.csv", index=False,
    )
    update_predictions.to_csv(
        session_dir / "unified_global_selected_model_user_update_predictions.csv", index=False,
    )

    attack_pool = joblib.load(
        root / "data/processed/fixed_multi_input_attack_pool_seed66.joblib"
    )
    attack_state_ids = np.asarray(score_payload["attack_state_ids"], dtype=int)
    attack_scores = np.asarray(score_payload["attack_scores"], dtype=float)
    attack_predictions = attack_scores >= float(score_payload["threshold"])
    origins = np.asarray(attack_pool["state_event_types"], dtype=str)[attack_state_ids]
    origin_rows = []
    for origin in ("activation", "user_update"):
        mask = origins == origin
        tp = int(attack_predictions[mask].sum())
        fn = int(mask.sum() - tp)
        tn = int(global_row["tn"])
        fp = int(global_row["fp"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        origin_rows.append({
            "model_id": global_model_id,
            "family": global_row["family"],
            "protocol": global_row["protocol"],
            "attack_origin": origin,
            "attack_count": int(mask.sum()),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "precision": precision, "recall": recall, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
        })
    pd.DataFrame(origin_rows).to_csv(
        session_dir / "unified_global_selected_attack_origin_metrics.csv", index=False,
    )
    payload = {
        "candidate_count": len(registry["configs"]),
        "cv_task_count": 5 * len(registry["configs"]),
        "selected_model_count": len(registry["selected"]),
        "configs": registry["configs"],
    }
    payload["registry_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    (session_dir / "unified_candidate_registry.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    (session_dir / "unified_global_selected_model.json").write_text(
        json.dumps(json.loads(selected.iloc[0].to_json()), indent=2), encoding="utf-8",
    )
    return selected, family, payload


def load_normal_inputs(root: Path, source_ids):
    root = Path(root).resolve()
    pool = joblib.load(root / "data/processed/fixed_multi_input_attack_pool_seed66.joblib")
    sources = np.asarray(pool["state_source_ids"], dtype=int)
    mask = np.isin(sources, np.asarray(source_ids, dtype=int))
    raw = json.loads((root / "data/raw/caltech_sessions_2021.json").read_text(encoding="utf-8"))
    records = [record for record in raw if (record.get("userInputs") or []) and float(record["userInputs"][0].get("kWhRequested", 0.0)) > 0]
    source_station = np.asarray([str(record.get("stationID", "UNKNOWN")) for record in records], dtype=object)
    profile = np.asarray(pool["normal_profile_features"], dtype=float)[mask]
    transition = np.asarray(pool["normal_transition_features"], dtype=float)[mask]
    is_change = np.asarray(pool["state_event_types"], dtype=str)[mask] == "user_update"
    previous = profile - transition[:, 5:10]
    compact = compact_transition(profile, previous)
    compact[~is_change] = 0.0
    return {
        "profile": profile,
        "transition": transition,
        "compact": compact,
        "is_change": is_change,
        "stations": source_station[sources[mask]],
        "event_types": np.asarray(pool["state_event_types"], dtype=str)[mask],
    }


def event_model_inputs(events: pd.DataFrame, root: Path):
    events = events.copy().reset_index(drop=True)
    profile = events[[f"x{index}" for index in range(5)]].to_numpy(dtype=float)
    previous = profile.copy()
    scenario_columns = ["split_role", "scenario_kind"]
    if "scenario_rep" in events.columns:
        scenario_columns.append("scenario_rep")
    session_columns = scenario_columns + ["source_session"]
    order = events.assign(_row=np.arange(len(events))).sort_values(
        session_columns + ["time", "event_type", "_row"]
    )
    for _, part in order.groupby(session_columns, sort=False):
        indices = part["_row"].to_numpy(dtype=int)
        if len(indices) > 1:
            previous[indices[1:]] = profile[indices[:-1]]
    is_change = ~events["event_type"].eq("activation").to_numpy()
    previous[~is_change] = profile[~is_change]
    delta = profile - previous
    delta[~is_change] = 0.0
    transition = np.column_stack([profile, delta, is_change.astype(float)])
    compact = compact_transition(profile, previous)
    compact[~is_change] = 0.0
    raw = json.loads((Path(root) / "data/raw/caltech_sessions_2021.json").read_text(encoding="utf-8"))
    records = [record for record in raw if (record.get("userInputs") or []) and float(record["userInputs"][0].get("kWhRequested", 0.0)) > 0]
    source_station = np.asarray([str(record.get("stationID", "UNKNOWN")) for record in records], dtype=object)
    stations = source_station[events["source_session"].to_numpy(dtype=int)]
    return {
        "profile": profile,
        "transition": transition,
        "compact": compact,
        "is_change": is_change,
        "stations": stations,
    }


def fit_and_score_selected_model(record, train_inputs, calibration_inputs, event_inputs, seed):
    config = record["config"]
    if record["model_block"] == "standard":
        scorer = fit_original_scorer(
            config,
            train_inputs["profile"], train_inputs["transition"], train_inputs["is_change"], seed,
        )
        calibration_scores = scorer(
            calibration_inputs["profile"], calibration_inputs["transition"], calibration_inputs["is_change"],
        )
        scores = scorer(event_inputs["profile"], event_inputs["transition"], event_inputs["is_change"])
    else:
        scorer = fit_masked_candidate(
            config,
            train_inputs["profile"], train_inputs["compact"], train_inputs["is_change"],
            train_inputs["stations"], seed,
        )
        calibration_scores = scorer(
            calibration_inputs["profile"], calibration_inputs["compact"], calibration_inputs["is_change"],
            calibration_inputs["stations"],
        )
        scores = scorer(
            event_inputs["profile"], event_inputs["compact"], event_inputs["is_change"], event_inputs["stations"],
        )
    return {
        **record,
        "calibration_scores": np.asarray(calibration_scores, dtype=float),
        "calibration_event_types": np.asarray(calibration_inputs["event_types"], dtype=str),
        "event_scores": np.asarray(scores, dtype=float),
    }
