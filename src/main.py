"""
alcnet-attention-layer-convolution | Training core

This module implements the core training loop for the hybrid model with a structure that can be orchestrated
by FastAPI + Celery + Redis and extended with NVTX/Nsight profiling later. The code includes an NVTX helper.
"""

from __future__ import annotations

import os
import json
import math
import time
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, set_seed as hf_set_seed
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm


# ---------------- Repro, device, and NVTX helpers ----------------
GLOBAL_SEED = 42


def set_all_seeds(seed: int = GLOBAL_SEED):
    hf_set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Favor performance while keeping reasonable determinism
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def pick_device(force: Optional[str] = None) -> torch.device:
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def effective_num_workers(requested: Optional[int]) -> int:
    # If inside Celery, prefer workers=0 for safety (no fork issues)
    if os.environ.get("IN_CELERY", "0") == "1":
        return 0
    return 0 if requested is None else max(0, int(requested))


class nvtx_range:
    def __init__(self, name: str, enabled: bool):
        self.name = name
        self.enabled = enabled and torch.cuda.is_available()

    def __enter__(self):
        if self.enabled:
            try:
                torch.cuda.nvtx.range_push(self.name)
            except Exception:
                pass

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass


# ---------------- Data ----------------
class RTEDataset(Dataset):
    def __init__(self, df, tokenizer, max_len: int):
        self.prem = df["sentence1"]
        self.hyp = df["sentence2"]
        self.labels = df["label"].astype(np.int64)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        p = self.prem.iloc[idx] if hasattr(self.prem, "iloc") else self.prem[idx]
        h = self.hyp.iloc[idx] if hasattr(self.hyp, "iloc") else self.hyp[idx]
        y = int(self.labels.iloc[idx]) if hasattr(self.labels, "iloc") else int(self.labels[idx])
        enc = self.tok(
            p,
            h,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(y, dtype=torch.long),
        }


def collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}


# ---------------- Encoder and attention utilities ----------------
class EncoderWrapper(nn.Module):
    def __init__(self, model_name: str, gradient_checkpointing: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=True,
            attn_implementation="eager",
        )
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            try:
                self.encoder.gradient_checkpointing_enable()
            except Exception:
                pass

    def forward(self, **enc):
        out = self.encoder(**enc)
        cls = out.last_hidden_state[:, 0]
        return out.attentions, cls


_MASK_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}


def _mask_far(S: int, device, k: int = 5):
    key = (S, f"far{k}")
    m = _MASK_CACHE.get(key)
    if m is None or m.device != device:
        idx = torch.arange(S, device=device)
        m = (idx[None, :] - idx[:, None]).abs() >= k
        _MASK_CACHE[key] = m
    return m


def _mask_diag(S: int, device):
    key = (S, "diag")
    m = _MASK_CACHE.get(key)
    if m is None or m.device != device:
        m = torch.eye(S, device=device).bool()
        _MASK_CACHE[key] = m
    return m


def _nan_to_num(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


@torch.no_grad()
def attn_stats_21(att_list: List[torch.Tensor], layers: List[int], k: int = 5) -> torch.Tensor:
    stats = []

    def layer_feats(A):
        den_mat = A.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        P = A / den_mat
        S = A.size(-1)
        far = _mask_far(S, A.device, k=k)
        lr = (A * far).sum(dim=(-2, -1)) / A.sum(dim=(-2, -1)).clamp_min(1e-8)
        cls_in = P[..., 0].mean(dim=-1)
        P_safe = P.clamp_min(1e-8)
        ent = -(P_safe * P_safe.log()).sum(dim=-1).mean(dim=-1)
        ent_std = ent.std(dim=-1)
        mrow = A.max(dim=-1).values.mean(dim=-1)
        denom = A.sum(dim=(-2, -1)).clamp_min(1e-8)
        mrow = mrow / denom
        asym = (A - A.transpose(-1, -2)).abs()
        asym = asym.sum(dim=(-2, -1)) / A.sum(dim=(-2, -1)).clamp_min(1e-8)
        scalars = [lr.mean(dim=-1), cls_in.mean(dim=-1), ent.mean(dim=-1), ent_std, mrow.mean(dim=-1), asym.mean(dim=-1)]
        return [_nan_to_num(s) for s in scalars]

    L = len(att_list)
    sel = [att_list[i] if i >= 0 else att_list[L + i] for i in layers]
    for A in sel:
        A = A.clamp_min(0) + 1e-12
        stats += layer_feats(A)

    A = sel[-1]
    S = A.size(-1)
    diag = _mask_diag(S, A.device)
    diag_mass = (A * diag).sum(dim=(-2, -1)) / A.sum(dim=(-2, -1)).clamp_min(1e-8)
    off_mass = 1.0 - diag_mass
    eff_heads = (off_mass > 0.60).float().mean(dim=-1)

    stats += [diag_mass.mean(dim=-1), off_mass.mean(dim=-1), eff_heads]
    out = torch.stack(stats, dim=1).float()
    return _nan_to_num(out)


def build_attn_tensor(attn_list: List[torch.Tensor], layer_idx: List[int], resize_to: int, channel_norm: bool = True) -> torch.Tensor:
    x = torch.cat([attn_list[i] for i in layer_idx], dim=1)  # (B, C, S, S)
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    if channel_norm:
        eps = 1e-5
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        x = (x - mean) / (std + eps)
    if x.shape[-1] != resize_to:
        x = F.interpolate(x, size=(resize_to, resize_to), mode="bilinear", align_corners=False)
    return x


# ---------------- Model ----------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, drop=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.dp = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.dp(x)
        return x


class AttnCNN(nn.Module):
    def __init__(self, in_ch: int, out_dim: int = 256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            ConvBlock(64, 128, drop=0.05),
            nn.MaxPool2d(2),
            ConvBlock(128, 256, drop=0.05),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.adapter(x)
        x = self.body(x).flatten(1)
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden = max(128, in_dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class GatedHybridClassifier(nn.Module):
    def __init__(self, attn_in_ch: int, cls_dim: int, stats_dim: int = 21, num_classes: int = 2):
        super().__init__()
        self.transformer_head = nn.Linear(cls_dim, num_classes)
        self.cnn = AttnCNN(attn_in_ch, out_dim=256)
        self.stats_mlp = MLP(stats_dim, out_dim=128)
        cnn_combined_dim = 256 + 128
        self.cnn_head = nn.Linear(cnn_combined_dim, num_classes)
        self.conf_head = nn.Linear(cnn_combined_dim, 1)
        nn.init.constant_(self.conf_head.bias, -2.0)

    def forward(self, attn_img, cls_vec, stats_vec):
        vanilla_logits = self.transformer_head(cls_vec)
        cnn_feat = self.cnn(attn_img)
        stats_feat = self.stats_mlp(stats_vec)
        combined = torch.cat([cnn_feat, stats_feat], dim=1)
        cnn_logits = self.cnn_head(combined)
        conf = torch.sigmoid(self.conf_head(combined))
        logits = (1 - conf) * vanilla_logits + conf * cnn_logits
        return logits


# ---------------- Config ----------------
@dataclass
class AlcnetCfg:
    model_name: str = "roberta-large"
    task_name: str = "rte"
    max_len: int = 128
    batch_size: int = 8
    epochs: int = 3
    lr_head: float = 3e-4
    lr_encoder: float = 8e-6
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    attn_layers: Tuple[int, int, int] = (-3, -2, -1)
    attn_size: int = 128
    seed: int = GLOBAL_SEED
    device: Optional[str] = None  # "cpu" | "cuda" | None(auto)
    # optional caps
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None
    # toggles
    gradient_checkpointing: bool = True


def parse_layers(sel: Tuple[int, ...], n_layers: int) -> List[int]:
    out: List[int] = []
    for i in sel:
        j = i if i >= 0 else (n_layers + i)
        if 0 <= j < n_layers:
            out.append(j)
    return sorted(set(out))


def build_loaders(cfg: AlcnetCfg, device: torch.device, dataloader_workers: Optional[int] = None):
    raw = load_dataset("glue", cfg.task_name)
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    train_pd = raw["train"].to_pandas()
    val_pd = raw["validation"].to_pandas()

    if cfg.max_train_samples is not None:
        train_pd = train_pd.sample(cfg.max_train_samples, random_state=cfg.seed).reset_index(drop=True)
    if cfg.max_val_samples is not None:
        val_pd = val_pd.sample(cfg.max_val_samples, random_state=cfg.seed).reset_index(drop=True)

    train_ds = RTEDataset(train_pd, tok, cfg.max_len)
    val_ds = RTEDataset(val_pd, tok, cfg.max_len)

    nw = effective_num_workers(dataloader_workers)
    pin = device.type == "cuda"
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        collate_fn=collate,
        drop_last=False,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        collate_fn=collate,
        drop_last=False,
    )
    return tok, train_dl, val_dl


@torch.no_grad()
def evaluate(encoder, model, dl, device, layer_idx, attn_size, max_batches=None) -> Tuple[float, float]:
    encoder.eval()
    model.eval()
    preds, gts = [], []
    for bi, batch in enumerate(dl):
        y = batch["label"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        att_list, cls_vec = encoder(input_ids=input_ids, attention_mask=attention_mask)
        X = build_attn_tensor(att_list, layer_idx, attn_size, True).to(device, non_blocking=True)
        S = attn_stats_21(att_list, layer_idx).to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(X, cls_vec, S)
        p = logits.argmax(dim=1)
        preds.append(p.cpu())
        gts.append(y.cpu())
        if max_batches and (bi + 1) >= max_batches:
            break

    preds = torch.cat(preds).numpy()
    gts = torch.cat(gts).numpy()
    return float(accuracy_score(gts, preds)), float(f1_score(gts, preds))


def train_and_eval(
    *,
    task_ctx: Optional[Any] = None,   # Celery task (for update_state), or None
    device: Optional[torch.device] = None,
    cfg: Optional[AlcnetCfg] = None,
    save_dir: Optional[str] = None,
    save_artifacts: bool = False,
    progress_path: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = cfg or AlcnetCfg()
    set_all_seeds(cfg.seed)
    device = device or pick_device(cfg.device)

    tok, train_dl, val_dl = build_loaders(cfg, device)
    encoder = EncoderWrapper(cfg.model_name, cfg.gradient_checkpointing).to(device)

    # Probe shapes for model init
    with torch.no_grad():
        sample = next(iter(train_dl))
        ids = sample["input_ids"][:2].to(device)
        msk = sample["attention_mask"][:2].to(device)
        att_probe, cls_probe = encoder(input_ids=ids, attention_mask=msk)
        n_layers = len(att_probe)
        n_heads = att_probe[0].shape[1]
        layer_idx = parse_layers(tuple(cfg.attn_layers), n_layers)
        in_ch = n_heads * len(layer_idx)
        cls_dim = int(cls_probe.shape[1])

    model = GatedHybridClassifier(attn_in_ch=in_ch, cls_dim=cls_dim, stats_dim=21, num_classes=2).to(device)

    enc_params = list(encoder.parameters())
    head_params = list(model.parameters())

    opt = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": cfg.lr_encoder, "weight_decay": cfg.weight_decay},
            {"params": head_params, "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
        ]
    )
    total_steps = max(1, math.ceil(len(train_dl)) * cfg.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    ce_loss = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    logs: List[Dict[str, Any]] = []
    best = {"val_acc": -1.0, "epoch": -1, "val_f1": None}
    step = 0

    for ep in range(1, cfg.epochs + 1):
        encoder.train(); model.train()
        start = time.time(); run_loss = 0.0; run_correct = 0; run_total = 0
        with nvtx_range(f"epoch_{ep}", enabled=(device.type == "cuda")):
            pbar = tqdm(train_dl, total=len(train_dl), desc=f"Epoch {ep}/{cfg.epochs}", dynamic_ncols=True, leave=False)
            for bi, batch in enumerate(pbar):
                y = batch["label"].to(device, non_blocking=True)
                ids = batch["input_ids"].to(device, non_blocking=True)
                msk = batch["attention_mask"].to(device, non_blocking=True)

                att_list, cls_vec = encoder(input_ids=ids, attention_mask=msk)
                X = build_attn_tensor(att_list, layer_idx, cfg.attn_size, True).to(device, non_blocking=True)
                S = attn_stats_21(att_list, layer_idx).to(device, non_blocking=True)

                with nvtx_range("forward", enabled=(device.type == "cuda")):
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                        logits = model(X, cls_vec, S)
                        loss = ce_loss(logits, y)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if cfg.grad_clip and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    if enc_params:
                        torch.nn.utils.clip_grad_norm_(enc_params, cfg.grad_clip)
                with nvtx_range("step", enabled=(device.type == "cuda")):
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                step += 1

                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                    correct = (pred == y).sum().item()
                    total = y.numel()
                    run_correct += correct; run_total += total; run_loss += float(loss.item()) * total
                    pbar.set_postfix_str(
                        f"loss={run_loss/max(1,run_total):.4f} | acc={100.0*run_correct/max(1,run_total):.2f}%"
                    )
                if cfg.max_train_batches and (bi + 1) >= cfg.max_train_batches:
                    break

        val_acc, val_f1 = evaluate(
            encoder, model, val_dl, device, layer_idx, cfg.attn_size, max_batches=cfg.max_val_batches
        )

        ep_row = {
            "epoch": ep,
            "time_sec": round(time.time() - start, 2),
            "train_acc": float(run_correct / max(1, run_total)),
            "train_loss_ema": float(run_loss / max(1, run_total)),
            "val_acc": float(val_acc),
            "val_f1_macro": float(val_f1),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        logs.append(ep_row)
        if progress_path:
            try:
                with open(progress_path, "a") as pf:
                    pf.write(json.dumps(ep_row) + "\n")
            except Exception:
                pass
        if task_ctx is not None:
            try:
                task_ctx.update_state(state="STARTED", meta={"progress_epoch": ep, "epochs_total": cfg.epochs, "last_val": {"acc": val_acc, "f1": val_f1}})
            except Exception:
                pass

        if val_acc > best["val_acc"]:
            best = {"val_acc": float(val_acc), "epoch": ep, "val_f1": float(val_f1)}
            if save_dir and save_artifacts:
                try:
                    torch.save(
                        {
                            "encoder": encoder.state_dict(),
                            "model": model.state_dict(),
                            "attn_layers": layer_idx,
                            "attn_size": cfg.attn_size,
                            "in_ch": in_ch,
                            "cls_dim": cls_dim,
                            "model_name": cfg.model_name,
                            "task_name": cfg.task_name,
                        },
                        os.path.join(save_dir, "model.pt"),
                    )
                except Exception:
                    pass

    return {
        "seed": cfg.seed,
        "device": str(device),
        "cfg": asdict(cfg),
        "best": best,
        "epochs": logs,
        "save_dir": save_dir if save_dir else None,
    }


__all__ = [
    "AlcnetCfg",
    "train_and_eval",
    "pick_device",
    "set_all_seeds",
]

