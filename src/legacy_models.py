"""Small supervised models with normal-only scaling and an unscaled change gate."""
from __future__ import annotations

import itertools
import random
import time

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn

from data_protocol import sample_weights

FAMILIES = ["Logistic Regression", "RBF SVC", "Random Forest", "Extra Trees", "XGBoost",
            "MLP", "Residual MLP", "AE + Supervised Head", "SDTN", "TabM"]
STABILITY = ["SDTN", "Residual MLP", "Extra Trees", "XGBoost", "Logistic Regression"]
BASELINES = ["Normal-only Dual-Branch", "Normal-only AE+OCSVM", "Normal-only Profile-Anchor"]


def registry():
    rows = []
    for family in FAMILIES:
        protocols = ["transition"] if family == "SDTN" else ["profile", "transition"]
        count = 24 if family == "SDTN" else 12
        for protocol in protocols:
            for j in range(count):
                params = {"width": [32, 64, 128][j % 3], "depth": 2 + (j // 3) % 2,
                          "dropout": [0.0, 0.15][(j // 6) % 2],
                          "learning_rate": [0.001, 0.003][(j // 3) % 2],
                          "weight_decay": [1e-5, 1e-3][(j // 6) % 2],
                          "epochs": 120, "embedding": [8, 16, 32][j % 3],
                          "mask_rate": 0.15, "aux_weight": 0.05 if j >= 12 else 0.0}
                if family == "Logistic Regression":
                    params = {"C": float(np.logspace(-4, 3, 12)[j])}
                elif family == "RBF SVC":
                    c, gamma = list(itertools.product([0.1, 1, 10, 100], ["scale", 0.1, 1.0]))[j]
                    params = {"C": c, "gamma": gamma}
                elif family in {"Random Forest", "Extra Trees"}:
                    depth, leaf = list(itertools.product([4, 8, None], [1, 2, 5, 10]))[j]
                    params = {"max_depth": depth, "min_samples_leaf": leaf, "n_estimators": 128}
                elif family == "XGBoost":
                    depth, lr, child = list(itertools.product([2, 3, 5], [0.03, 0.1], [1, 5]))[j]
                    params = {"max_depth": depth, "learning_rate": lr,
                              "min_child_weight": child, "n_estimators": 160,
                              "subsample": .8, "colsample_bytree": .9}
                rows.append({"config_id": f"c{len(rows):03d}", "family": family,
                             "protocol": protocol, "params": params})
    for family in BASELINES:
        if family == "Normal-only Dual-Branch":
            combinations = itertools.product([([32, 16], 3), ([64, 32], 4)], [.15, .30], ["scale", 1.0], [.05, .15, .30])
            items = [{"hidden": hidden, "latent": latent, "mask_rate": mask, "gamma": gamma,
                      "transition_weight": weight, "nu": .05, "epochs": 180, "transition_epochs": 160}
                     for (hidden, latent), mask, gamma, weight in combinations]
            protocols = ["transition"] * 24
        elif family == "Normal-only AE+OCSVM":
            combinations = list(itertools.product([([64, 32], 3), ([128, 64], 4)], [.025, .05, .1], ["scale", 1.0], ["profile", "transition"]))
            items = [{"hidden": h, "latent": z, "nu": nu, "gamma": gamma, "epochs": 180, "lr": .001}
                     for (h, z), nu, gamma, protocol in combinations]
            protocols = [p for _, _, _, p in combinations]
        else:
            combinations = itertools.product(["mahalanobis", "ocsvm"], [.05, .1, .2], ["scale", 1.0], [.15, .30])
            items = [{"transition_head": head, "nu": nu, "gamma": gamma, "transition_weight": w, "transition_nu": .1}
                     for head, nu, gamma, w in combinations]
            protocols = ["transition"] * 24
        for params, protocol in zip(items, protocols):
            rows.append({"config_id": f"c{len(rows):03d}", "family": family,
                         "protocol": protocol, "params": params})
    for record in rows:
        p = record["params"]
        dimension = 5 if record["protocol"] == "profile" else 11
        if "n_estimators" in p:
            complexity = p["n_estimators"] * 2 ** (p.get("max_depth") or 12) / p.get("min_samples_leaf", 1)
        elif "width" in p:
            complexity = p["width"] ** 2 * p["depth"]
        elif "hidden" in p:
            complexity = sum(p["hidden"]) ** 2
        else:
            complexity = dimension
        record["complexity_proxy"] = float(complexity)
    assert all(sum(c["family"] == f for c in rows) == 24 for f in FAMILIES + BASELINES)
    return rows


def block(a, b, dropout):
    return nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout))


class Dense(nn.Module):
    def __init__(self, dimension, classes, params, residual=False):
        super().__init__()
        width, depth, drop = params["width"], params["depth"], params["dropout"]
        self.input = block(dimension, width, drop)
        self.layers = nn.ModuleList([block(width, width, drop) for _ in range(depth - 1)])
        self.output = nn.Linear(width, classes)
        self.residual = residual

    def forward(self, x):
        x = self.input(x)
        for layer in self.layers:
            x = x + layer(x) if self.residual else layer(x)
        return self.output(x)


class Dual(nn.Module):
    def __init__(self, classes, p, ablation=None):
        super().__init__()
        width, embed, drop = p["width"], p["embedding"], p["dropout"]
        self.state = nn.Sequential(block(5, width, drop), block(width, width // 2, drop), nn.Linear(width // 2, embed))
        self.transition = nn.Sequential(block(5, width // 2, drop), block(width // 2, embed, drop), nn.Linear(embed, embed))
        self.state_decoder = nn.Sequential(block(embed, width // 2, 0), block(width // 2, width, 0), nn.Linear(width, 5))
        self.transition_decoder = nn.Sequential(block(embed, embed, 0), block(embed, width // 2, 0), nn.Linear(width // 2, 5))
        self.head = nn.Sequential(block(3 * embed + 1, width // 2, drop), nn.Linear(width // 2, classes))
        self.ablation = ablation

    def forward(self, x):
        hs, hd = self.state(x[:, :5]), self.transition(x[:, 5:10])
        gate = x[:, 10:11]
        assert torch.all((gate == 0) | (gate == 1)), "Change gate was standardized"
        if self.ablation == "state_only":
            gate = torch.zeros_like(gate)
        if self.ablation == "transition_only":
            hs = torch.zeros_like(hs)
        interaction = hs * hd
        if self.ablation == "no_interaction":
            interaction = torch.zeros_like(interaction)
        return self.head(torch.cat([hs, gate * hd, gate * interaction, gate], dim=1))

    def auxiliary(self, normal, updates, mask_rate):
        result = normal.sum() * 0
        for values, encoder, decoder in [(normal[:, :5], self.state, self.state_decoder),
                                        (updates[:, 5:10], self.transition, self.transition_decoder)]:
            if not len(values):
                continue
            mask = torch.rand_like(values) < mask_rate
            error = (decoder(encoder(values.masked_fill(mask, 0))) - values).square()
            result = result + (error * mask).sum() / mask.sum().clamp_min(1)
        return result


class Autoencoder(nn.Module):
    def __init__(self, dimension, width, latent):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dimension, width), nn.GELU(), nn.Linear(width, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, width), nn.GELU(), nn.Linear(width, dimension))


class Fitted:
    def __init__(self, config, seed, task="binary", ablation=None):
        self.config, self.seed, self.task, self.ablation = config, int(seed), task, ablation
        self.classes = 2 if task == "binary" else 7

    def matrix(self, x):
        x = np.asarray(x, dtype=np.float32)
        width = 5 if self.config["protocol"] == "profile" else 10
        values = self.scaler.transform(x[:, :width]).astype(np.float32)
        if width == 10 and self.ablation != "no_indicator":
            values = np.column_stack([values, x[:, 10]])
        return values

    def fit(self, data, types):
        started = time.perf_counter()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        if self.config["family"] in BASELINES:
            self.fit_normal_only(data)
            self.seconds = time.perf_counter() - started
            return self
        width = 5 if self.config["protocol"] == "profile" else 10
        self.scaler = StandardScaler().fit(data.x[data.y == 0, :width])
        x = self.matrix(data.x)
        target = data.y if self.task == "binary" else data.mechanism
        w = sample_weights(data, types)
        if self.ablation == "label_permutation":
            target = np.random.default_rng(self.seed).permutation(target)
        family, p = self.config["family"], self.config["params"]
        if family == "Logistic Regression":
            self.model = LogisticRegression(**p, max_iter=2000)
        elif family == "RBF SVC":
            self.model = SVC(**p)
        elif family in {"Random Forest", "Extra Trees"}:
            cls = RandomForestClassifier if family == "Random Forest" else ExtraTreesClassifier
            self.model = cls(**p, n_jobs=1, random_state=self.seed)
        elif family == "XGBoost":
            from xgboost import XGBClassifier
            self.model = XGBClassifier(**p, tree_method="hist", n_jobs=1, random_state=self.seed,
                                       reg_lambda=1.0, verbosity=0)
        else:
            self.fit_neural(x, target, w, data, family, p)
            self.seconds = time.perf_counter() - started
            return self
        self.model.fit(x, target, sample_weight=w)
        self.neural = False
        self.seconds = time.perf_counter() - started
        return self

    @staticmethod
    def original_inputs(x):
        order = [0, 4, 2, 3, 1]
        profile = x[:, order]
        transition = np.column_stack([profile, x[:, np.array(order) + 5], x[:, 10]])
        return profile, transition

    def fit_normal_only(self, data):
        from masked_transition_model_search_core import compact_transition, fit_candidate
        from session_model_runtime import fit_original_scorer
        normal = data.take(data.y == 0)
        profile, transition = self.original_inputs(normal.x)
        p = dict(self.config["params"])
        if self.config["family"] == "Normal-only AE+OCSVM":
            config = {"algorithm": "ae_ocsvm", "protocol": "profile_only" if self.config["protocol"] == "profile" else "transition_aware", "params": p}
            self.normal_head = fit_original_scorer(config, profile, transition, normal.update, self.seed)
            self.original_kind = "original"
        else:
            compact = compact_transition(profile, profile - transition[:, 5:10])
            self.original_kind = "masked"
            config = {"state_head": "masked_ae_ocsvm" if self.config["family"] == "Normal-only Dual-Branch" else "ocsvm",
                      "transition_head": "masked_ae" if self.config["family"] == "Normal-only Dual-Branch" else p.pop("transition_head"),
                      "params": p}
            self.normal_head = fit_candidate(config, profile, compact, normal.update, np.full(len(profile), "POOLED"), self.seed)

    def normal_score(self, raw):
        from masked_transition_model_search_core import compact_transition
        profile, transition = self.original_inputs(np.asarray(raw, float))
        if self.original_kind == "original":
            return self.normal_head(profile, transition, raw[:, 10].astype(bool))
        compact = compact_transition(profile, profile - transition[:, 5:10])
        return self.normal_head(profile, compact, raw[:, 10].astype(bool), np.full(len(profile), "POOLED"))

    def fit_neural(self, x, target, weights, data, family, p):
        self.neural = True
        x = torch.from_numpy(x)
        y = torch.from_numpy(np.asarray(target, dtype=np.int64))
        w = torch.from_numpy(np.asarray(weights, dtype=np.float32))
        normal = x[torch.from_numpy(data.y == 0)]
        updates = x[torch.from_numpy(data.update)]
        self.ae = None
        if family == "AE + Supervised Head":
            self.ae = Autoencoder(x.shape[1], p["width"], p["embedding"])
            optimizer = torch.optim.AdamW(self.ae.parameters(), lr=0.001)
            self.ae.train()
            for _ in range(80):
                optimizer.zero_grad()
                loss = (self.ae.decoder(self.ae.encoder(normal)) - normal).square().mean()
                loss.backward()
                optimizer.step()
            self.ae.eval()
            with torch.no_grad():
                x = self.representation(x)
            self.model = Dense(x.shape[1], self.classes, p)
        elif family == "SDTN":
            self.model = Dual(self.classes, p, self.ablation)
        elif family == "TabM":
            from tabm import TabM
            self.model = TabM.make(n_num_features=x.shape[1], d_out=self.classes, n_blocks=p["depth"],
                                   d_block=p["width"], dropout=p["dropout"], k=8)
        else:
            self.model = Dense(x.shape[1], self.classes, p, residual=family == "Residual MLP")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
        self.model.train()
        for _ in range(p["epochs"]):
            order = torch.randperm(len(x))
            for start in range(0, len(x), 1024):
                index = order[start:start + 1024]
                optimizer.zero_grad()
                logits = self.model(x[index])
                if logits.ndim == 3:
                    # TabM members are trained separately; only inference averages probabilities.
                    loss = nn.functional.cross_entropy(logits.flatten(0, 1), y[index].repeat_interleave(logits.shape[1]), reduction="none")
                    loss = loss.reshape(len(index), -1).mean(dim=1)
                else:
                    loss = nn.functional.cross_entropy(logits, y[index], reduction="none")
                objective = (loss * w[index]).sum() / w[index].sum()
                if family == "SDTN" and p["aux_weight"] and self.ablation != "no_reconstruction":
                    objective = objective + p["aux_weight"] * self.model.auxiliary(normal, updates, p["mask_rate"])
                objective.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()
        self.model.eval()

    def representation(self, tensor):
        code = self.ae.encoder(tensor)
        error = (tensor - self.ae.decoder(code)).square().mean(dim=1, keepdim=True)
        return torch.cat([code, error], dim=1)

    def predict(self, raw):
        if hasattr(self, "normal_head"):
            from scipy.special import expit
            positive = expit(self.normal_score(raw))
            return np.column_stack([1 - positive, positive])
        x = self.matrix(raw)
        if self.neural:
            predictions = []
            with torch.no_grad():
                for start in range(0, len(x), 2048):
                    tensor = torch.from_numpy(x[start:start + 2048])
                    if self.ae is not None:
                        tensor = self.representation(tensor)
                    logits = self.model(tensor)
                    p = torch.softmax(logits, dim=-1, dtype=torch.float64)
                    if p.ndim == 3:
                        p = p.mean(dim=1)
                    predictions.append(p.numpy())
            probability = np.concatenate(predictions)
        elif hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(x)
        else:
            from scipy.special import expit, softmax
            score = self.model.decision_function(x)
            probability = np.column_stack([1 - expit(score), expit(score)]) if score.ndim == 1 else softmax(score, axis=1)
        # Weighted-loss scores are ranking evidence, not claims of field prevalence calibration.
        return probability

    def score(self, raw):
        if hasattr(self, "normal_head"):
            return np.asarray(self.normal_score(raw), dtype=np.float64)
        if not self.neural and self.task == "binary" and hasattr(self.model, "decision_function"):
            return np.asarray(self.model.decision_function(self.matrix(raw)), dtype=np.float64)
        return np.sum(self.predict(raw)[:, 1:], axis=1, dtype=np.float64)
