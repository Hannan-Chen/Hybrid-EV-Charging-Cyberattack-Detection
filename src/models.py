"""Small matched networks and normal-only pretraining, CPU deterministic seeds."""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn

from data_protocol import sample_weights
from legacy_models import Dual, Fitted as OriginalFitted
from protocol import U_CONFIG, signature


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def parameters(c):
    return {"width": c["width"], "embedding": c["embedding"], "dropout": 0.}


class SADNet(nn.Module):
    def __init__(self, width, embedding):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(11, width, bias=False), nn.GELU(),
                                     nn.Linear(width, width//2, bias=False), nn.GELU(),
                                     nn.Linear(width//2, embedding, bias=False))
        self.decoder = nn.Sequential(nn.Linear(embedding, width//2), nn.GELU(),
                                     nn.Linear(width//2, width), nn.GELU(), nn.Linear(width, 10))


def dual_codes(model, values):
    return model.state(values[:, :5]), model.transition(values[:, 5:10])


def retention(model, teacher, normal, updates):
    hs, _ = dual_codes(model, normal)
    ts, _ = dual_codes(teacher, normal)
    loss = (hs-ts.detach()).square().mean()
    if len(updates):
        hd = model.transition(updates[:, 5:10])
        td = teacher.transition(updates[:, 5:10])
        loss = loss + (hd-td.detach()).square().mean()
    return loss


def robust_scale(reference):
    reference = np.asarray(reference, float)
    median = float(np.median(reference))
    q25, q75 = np.quantile(reference, [.25, .75])
    scale = max(float(q75-q25), 1e-6, .01*float(reference.std()))
    return median, scale


class Fitted:
    def __init__(self, config, seed):
        self.config, self.seed = copy.deepcopy(config), int(seed)

    def matrix(self, raw):
        raw = np.asarray(raw, float)
        return np.column_stack([self.scaler.transform(raw[:, :10]), raw[:, 10]]).astype(np.float32)

    def fit(self, data, root, ablation=None):
        seed_all(self.seed)
        t = time.perf_counter()
        c, family = self.config, self.config["family"]
        self.scaler = StandardScaler().fit(data.x[data.y == 0, :10])
        x = self.matrix(data.x)
        if family == "RF":
            self.model = RandomForestClassifier(n_estimators=c["trees"], max_depth=c["depth"],
                                               min_samples_leaf=c["leaf"], random_state=self.seed, n_jobs=1)
            self.model.fit(x, data.y, sample_weight=sample_weights(data))
            self.pretrain_seconds = 0.
        else:
            self.fit_network(data, torch.from_numpy(x), root, ablation)
        self.seconds = time.perf_counter()-t
        self.parameter_count = (sum(p.numel() for p in self.model.parameters()) if family != "RF"
                                else sum(tree.tree_.node_count for tree in self.model.estimators_))
        return self

    def pretrain(self, normal, updates, root):
        c = self.config
        kind = "sad" if c["family"] == "DeepSAD" else "dual"
        key = signature({"kind": kind, "width": c["width"], "wd": c["weight_decay"],
                         "seed": self.seed, "epochs": c["pretrain_epochs"],
                         "normal_matrix_sha": __import__("hashlib").sha256(normal.numpy().tobytes()).hexdigest()})
        path = Path(root)/"results/cache/pretrained"/(key+".pt")
        self.model = (SADNet(c["width"], c["embedding"]) if kind == "sad" else Dual(2, parameters(c)))
        if path.exists():
            cached = torch.load(path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(cached["state"])
            self.pretrain_seconds = cached["seconds"]
            self.pretrain_key = key
            return
        start = time.perf_counter()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001, weight_decay=c["weight_decay"])
        self.model.train()
        # Separate activation/update terms prevent scarce real changes being drowned out.
        for _ in range(c["pretrain_epochs"]):
            optimizer.zero_grad()
            if kind == "dual":
                loss = self.model.auxiliary(normal, updates, c["mask_rate"])
            else:
                mask = torch.rand_like(normal[:, :10]) < c["mask_rate"]
                corrupt = normal.clone()
                corrupt[:, :10] = corrupt[:, :10].masked_fill(mask, 0)
                rec = self.model.decoder(self.model.encoder(corrupt))
                loss = (rec-normal[:, :10]).square().mean()
                if len(updates):
                    loss = loss + (self.model.decoder(self.model.encoder(updates))-updates[:, :10]).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            optimizer.step()
        self.pretrain_seconds = time.perf_counter()-start
        self.pretrain_key = key
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+f".{__import__('os').getpid()}.tmp")
        torch.save({"state": self.model.state_dict(), "seconds": self.pretrain_seconds}, temporary)
        temporary.replace(path)

    def fit_network(self, data, x, root, ablation):
        c, family = self.config, self.config["family"]
        normal = x[torch.from_numpy(data.y == 0)]
        updates = x[torch.from_numpy(data.update)]
        if family in ["SS", "DeepSAD"]:
            self.pretrain(normal, updates, root)
        else:
            self.model = Dual(2, parameters(c))
            self.pretrain_seconds = 0.
        seed_all(self.seed + 100000)
        teacher = copy.deepcopy(self.model).eval() if family == "SS" else None
        if teacher is not None:
            for p in teacher.parameters():
                p.requires_grad_(False)
        if family == "DeepSAD":
            self.model.eval()
            with torch.no_grad():
                center = self.model.encoder(normal).mean(dim=0)
                # Follow Deep SAD's fixed-center safeguard against trivial zero maps.
                center = torch.where(center.abs() < .1, torch.where(center < 0, -.1, .1), center)
            self.center = center
        target = torch.from_numpy(data.y.astype(np.int64))
        weight = torch.from_numpy(sample_weights(data).astype(np.float32))
        weight = weight/weight.sum()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        self.model.train()
        for _ in range(c["epochs"]):
            optimizer.zero_grad()
            if family == "DeepSAD":
                distance = (self.model.encoder(x)-self.center).square().sum(dim=1)
                point_loss = torch.where(target == 0, distance, c["extra"]/(distance+1e-6))
                loss = (point_loss*weight).sum()
            else:
                logits = self.model(x)
                loss = (nn.functional.cross_entropy(logits, target, reduction="none")*weight).sum()
                if family == "SS":
                    if ablation != "no_reconstruction":
                        loss = loss + self.model.auxiliary(normal, updates, c["mask_rate"])
                    if ablation != "no_retention":
                        loss = loss + c["extra"]*retention(self.model, teacher, normal, updates)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective")
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            optimizer.step()
        self.model.eval()
        self.final_loss = float(loss.detach())

    def score(self, raw):
        x = self.matrix(raw)
        family = self.config["family"]
        if family == "RF":
            return self.model.predict_proba(x)[:, 1].astype(float)
        with torch.no_grad():
            values = torch.from_numpy(x)
            if family == "DeepSAD":
                result = (self.model.encoder(values)-self.center).square().sum(dim=1)
            else:
                logits = self.model(values)
                result = logits[:, 1]-logits[:, 0]
        return result.numpy().astype(float)


def fit_u(normal, seed):
    started = time.perf_counter()
    model = OriginalFitted(U_CONFIG, seed).fit(normal, [])
    model.seconds = time.perf_counter()-started
    return model
