"""
alcnet-attention-layer-convolution | Training core

This module implements the core training loop for the hybrid model with a structure that can be orchestrated
by FastAPI + Celery + Redis and extended with NVTX/Nsight profiling later. The code includes an NVTX helper.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers import set_seed as hf_set_seed

logger = logging.getLogger(__name__)

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


def pick_device(force: str | None = None) -> torch.device:
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def effective_num_workers(requested: int | None) -> int:
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
            with contextlib.suppress(Exception):
                torch.cuda.nvtx.range_push(self.name)

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            with contextlib.suppress(Exception):
                torch.cuda.nvtx.range_pop()


# ---------------- Data ----------------
class RTEDataset(Dataset):
    """Pre-tokenized RTE dataset. Tokenization happens once at init, not per __getitem__."""

    def __init__(self, df, tokenizer, max_len: int):
        premises = list(df["sentence1"])
        hypotheses = list(df["sentence2"])
        self.labels = torch.tensor(df["label"].astype(np.int64).values, dtype=torch.long)

        # Batch-tokenize the entire dataset once
        enc = tokenizer(
            premises,
            hypotheses,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
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
            with contextlib.suppress(Exception):
                self.encoder.gradient_checkpointing_enable()

    def forward(self, **enc):
        out = self.encoder(**enc)
        cls = out.last_hidden_state[:, 0]
        return out.attentions, cls


_MASK_CACHE: dict[tuple[int, str], torch.Tensor] = {}


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
def attn_stats_21(att_list: list[torch.Tensor], layers: list[int], k: int = 5) -> torch.Tensor:
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


def build_attn_tensor(attn_list: list[torch.Tensor], layer_idx: list[int], resize_to: int, channel_norm: bool = True) -> torch.Tensor:
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


# ---------------- FiLM Conditioning Layer ----------------
class FiLMGenerator(nn.Module):
    """Generate FiLM (gamma, beta) parameters from a conditioning vector.

    Given a conditioning input of shape (B, cond_dim), produces
    gamma and beta of shape (B, n_channels) for feature-wise affine
    transformation: out = gamma * features + beta.

    Reference: Perez et al., "FiLM: Visual Reasoning with a General
    Conditioning Layer", AAAI 2018.
    """

    def __init__(self, cond_dim: int, n_channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, n_channels * 2)
        # Initialize gamma near 1 and beta near 0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.proj.bias.data[:n_channels] = 1.0  # gamma init

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gb = self.proj(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta


def apply_film(features: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Apply FiLM conditioning: gamma * features + beta.

    features: (B, C, H, W) or (B, C)
    gamma, beta: (B, C)
    """
    if features.dim() == 4:
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
    return gamma * features + beta


# ---------------- Model ----------------
class DropPath(nn.Module):
    """Stochastic Depth per sample (Huang et al., 2016).

    During training, randomly drops the entire residual branch with probability
    `drop_prob`. At eval, acts as identity.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, dtype=x.dtype, device=x.device) + keep).floor_()
        return x * mask / keep


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, drop=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()
        self.dp = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x, gamma=None, beta=None):
        x = self.bn(self.conv(x))
        if gamma is not None and beta is not None:
            x = apply_film(x, gamma, beta)
        x = self.act(x)
        x = self.dp(x)
        return x


class AttnCNN(nn.Module):
    def __init__(self, in_ch: int, out_dim: int = 256, drop_path: float = 0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        # Unpack blocks for FiLM injection at each stage
        self.conv1 = ConvBlock(64, 128, drop=0.05)
        self.se1 = SEBlock(128)
        self.shortcut1 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.drop_path1 = DropPath(drop_path)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(128, 256, drop=0.05)
        self.se2 = SEBlock(256)
        self.shortcut2 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.drop_path2 = DropPath(drop_path)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.pool_gate = nn.Parameter(torch.tensor(0.5))  # learnable avg/max blend
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x, film_params=None):
        """Forward pass with optional FiLM conditioning.

        film_params: dict with 'gamma1','beta1','gamma2','beta2' or None
        """
        x = self.adapter(x)
        g1, b1 = (film_params["gamma1"], film_params["beta1"]) if film_params else (None, None)
        g2, b2 = (film_params["gamma2"], film_params["beta2"]) if film_params else (None, None)
        x = self.drop_path1(self.se1(self.conv1(x, g1, b1))) + self.shortcut1(x)
        x = self.pool(x)
        x = self.drop_path2(self.se2(self.conv2(x, g2, b2))) + self.shortcut2(x)
        g = torch.sigmoid(self.pool_gate)
        x = (g * self.avg_pool(x) + (1 - g) * self.max_pool(x)).flatten(1)
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden = max(128, in_dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ModelEMA:
    """Exponential moving average of model parameters.

    Maintains a shadow copy of model weights: shadow = decay * shadow + (1-decay) * param.
    Use context manager for evaluation with EMA weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.model = model

    @torch.no_grad()
    def update(self) -> None:
        for k, v in self.model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    @contextlib.contextmanager
    def apply(self):
        """Context manager: temporarily swap in EMA weights for eval."""
        original = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow)
        try:
            yield
        finally:
            self.model.load_state_dict(original)


def _init_weights(m: nn.Module) -> None:
    """Kaiming init for conv/linear with GELU, Xavier for classification heads."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class GatedHybridClassifier(nn.Module):
    def __init__(self, attn_in_ch: int, cls_dim: int, stats_dim: int = 21, num_classes: int = 2):
        super().__init__()
        self.transformer_head = nn.Linear(cls_dim, num_classes)
        self.cnn = AttnCNN(attn_in_ch, out_dim=256)
        self.stats_mlp = MLP(stats_dim, out_dim=128)

        # FiLM generators: stats → conditioning params for each CNN stage
        self.film1 = FiLMGenerator(stats_dim, n_channels=128)  # conv1 output channels
        self.film2 = FiLMGenerator(stats_dim, n_channels=256)  # conv2 output channels

        cnn_combined_dim = 256 + 128
        self.cnn_head = nn.Linear(cnn_combined_dim, num_classes)
        self.conf_head = nn.Linear(cnn_combined_dim, 1)
        nn.init.constant_(self.conf_head.bias, -2.0)

        # Apply init to CNN/MLP submodules, then Xavier for classification heads
        self.cnn.apply(_init_weights)
        self.stats_mlp.apply(_init_weights)
        nn.init.xavier_uniform_(self.transformer_head.weight)
        nn.init.xavier_uniform_(self.cnn_head.weight)

    def forward(self, attn_img, cls_vec, stats_vec, ablation: str | None = None):
        vanilla_logits = self.transformer_head(cls_vec)

        if ablation == "no_cnn":
            return vanilla_logits

        # Generate FiLM parameters from stats
        use_film = ablation != "no_film"
        if use_film:
            gamma1, beta1 = self.film1(stats_vec)
            gamma2, beta2 = self.film2(stats_vec)
            film_params = {
                "gamma1": gamma1, "beta1": beta1,
                "gamma2": gamma2, "beta2": beta2,
            }
        else:
            film_params = None

        cnn_feat = self.cnn(attn_img, film_params=film_params)

        if ablation == "no_stats":
            stats_feat = torch.zeros(stats_vec.shape[0], 128, device=stats_vec.device)
        else:
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
    rdrop_alpha: float = 0.0  # R-Drop KL weight; >0 enables R-Drop (Liang et al., 2021)
    grad_clip: float = 1.0
    grad_accumulation_steps: int = 1
    attn_layers: tuple[int, int, int] = (-3, -2, -1)
    attn_size: int = 128
    seed: int = GLOBAL_SEED
    device: str | None = None  # "cpu" | "cuda" | None(auto)
    # optional caps
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    # schedule
    warmup_ratio: float = 0.06  # fraction of total steps for LR warmup
    lr_schedule: str = "cosine"  # "cosine" | "cosine_restarts"
    early_stopping_patience: int = 0  # 0 = disabled; N > 0 = stop after N epochs without improvement
    # toggles
    gradient_checkpointing: bool = True
    compile_model: bool = False  # torch.compile() — requires PyTorch 2.0+
    ema_decay: float = 0.0  # EMA of model weights; >0 enables (e.g. 0.999)
    ablation: str | None = None  # "no_cnn" | "no_stats" | "no_film" | None


def parse_layers(sel: tuple[int, ...], n_layers: int) -> list[int]:
    out: list[int] = []
    for i in sel:
        j = i if i >= 0 else (n_layers + i)
        if 0 <= j < n_layers:
            out.append(j)
    return sorted(set(out))


def build_loaders(cfg: AlcnetCfg, device: torch.device, dataloader_workers: int | None = None):
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
def evaluate(encoder, model, dl, device, layer_idx, attn_size, max_batches=None, ablation: str | None = None) -> tuple[float, float, float]:
    """Returns (accuracy, f1_macro, val_loss)."""
    encoder.eval()
    model.eval()
    ce = nn.CrossEntropyLoss()
    preds, gts, losses = [], [], []
    for bi, batch in enumerate(dl):
        y = batch["label"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        att_list, cls_vec = encoder(input_ids=input_ids, attention_mask=attention_mask)
        X = build_attn_tensor(att_list, layer_idx, attn_size, True).to(device, non_blocking=True)
        S = attn_stats_21(att_list, layer_idx).to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(X, cls_vec, S, ablation=ablation)
        losses.append(ce(logits.float(), y).item() * y.numel())
        p = logits.argmax(dim=1)
        preds.append(p.cpu())
        gts.append(y.cpu())
        if max_batches and (bi + 1) >= max_batches:
            break

    preds = torch.cat(preds).numpy()
    gts = torch.cat(gts).numpy()
    total = len(gts)
    val_loss = sum(losses) / max(1, total)
    return float(accuracy_score(gts, preds)), float(f1_score(gts, preds)), float(val_loss)


def train_and_eval(
    *,
    task_ctx: Any | None = None,   # Celery task (for update_state), or None
    device: torch.device | None = None,
    cfg: AlcnetCfg | None = None,
    save_dir: str | None = None,
    save_artifacts: bool = False,
    progress_path: str | None = None,
) -> dict[str, Any]:
    cfg = cfg or AlcnetCfg()
    set_all_seeds(cfg.seed)
    device = device or pick_device(cfg.device)

    _tok, train_dl, val_dl = build_loaders(cfg, device)
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

    if cfg.compile_model and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile()")
        model = torch.compile(model)

    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.ema_decay > 0 else None

    enc_params = list(encoder.parameters())
    head_params = list(model.parameters())

    opt = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": cfg.lr_encoder, "weight_decay": cfg.weight_decay},
            {"params": head_params, "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
        ]
    )
    total_steps = max(1, math.ceil(len(train_dl)) * cfg.epochs)
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

    post_warmup = max(1, total_steps - warmup_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        t = current_step - warmup_steps
        if cfg.lr_schedule == "cosine_restarts":
            # Cosine with warm restarts (SGDR, Loshchilov & Hutter 2017)
            # Restart period = steps_per_epoch
            steps_per_epoch = max(1, math.ceil(len(train_dl)))
            cycle_pos = t % steps_per_epoch
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * cycle_pos / steps_per_epoch)))
        # Default: single cosine decay
        progress = float(t) / float(post_warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    logs: list[dict[str, Any]] = []
    best = {"val_acc": -1.0, "epoch": -1, "val_f1": None}
    step = 0
    patience_counter = 0

    for ep in range(1, cfg.epochs + 1):
        encoder.train()
        model.train()
        start = time.time()
        run_loss = 0.0
        run_correct = 0
        run_total = 0
        with nvtx_range(f"epoch_{ep}", enabled=(device.type == "cuda")):
            pbar = tqdm(train_dl, total=len(train_dl), desc=f"Epoch {ep}/{cfg.epochs}", dynamic_ncols=True, leave=False)
            for bi, batch in enumerate(pbar):
                y = batch["label"].to(device, non_blocking=True)
                ids = batch["input_ids"].to(device, non_blocking=True)
                msk = batch["attention_mask"].to(device, non_blocking=True)

                att_list, cls_vec = encoder(input_ids=ids, attention_mask=msk)
                X = build_attn_tensor(att_list, layer_idx, cfg.attn_size, True).to(device, non_blocking=True)
                S = attn_stats_21(att_list, layer_idx).to(device, non_blocking=True)

                with nvtx_range("forward", enabled=(device.type == "cuda")), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(X, cls_vec, S, ablation=cfg.ablation)
                    loss = ce_loss(logits, y)

                    # R-Drop: second forward pass with different dropout, KL between both
                    if cfg.rdrop_alpha > 0:
                        logits2 = model(X, cls_vec, S)
                        loss2 = ce_loss(logits2, y)
                        p = F.log_softmax(logits, dim=-1)
                        q = F.log_softmax(logits2, dim=-1)
                        kl = 0.5 * (
                            F.kl_div(p, q.exp(), reduction="batchmean")
                            + F.kl_div(q, p.exp(), reduction="batchmean")
                        )
                        loss = 0.5 * (loss + loss2) + cfg.rdrop_alpha * kl

                    if cfg.grad_accumulation_steps > 1:
                        loss = loss / cfg.grad_accumulation_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("NaN/Inf loss at epoch %d batch %d — skipping", ep, bi)
                    opt.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()

                is_accumulation_step = (bi + 1) % cfg.grad_accumulation_steps == 0 or (bi + 1) == len(train_dl)
                if is_accumulation_step:
                    scaler.unscale_(opt)
                    if cfg.grad_clip and cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        if enc_params:
                            torch.nn.utils.clip_grad_norm_(enc_params, cfg.grad_clip)
                    with nvtx_range("step", enabled=(device.type == "cuda")):
                        scaler.step(opt)
                        scaler.update()
                        opt.zero_grad(set_to_none=True)
                        sched.step()
                    if ema is not None:
                        ema.update()
                    step += 1

                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                    correct = (pred == y).sum().item()
                    total = y.numel()
                    run_correct += correct
                    run_total += total
                    run_loss += float(loss.item()) * total
                    pbar.set_postfix_str(
                        f"loss={run_loss/max(1,run_total):.4f} | acc={100.0*run_correct/max(1,run_total):.2f}%"
                    )
                if cfg.max_train_batches and (bi + 1) >= cfg.max_train_batches:
                    break

        ema_ctx = ema.apply() if ema is not None else contextlib.nullcontext()
        with ema_ctx:
            val_acc, val_f1, val_loss = evaluate(
                encoder, model, val_dl, device, layer_idx, cfg.attn_size,
                max_batches=cfg.max_val_batches, ablation=cfg.ablation,
            )

        ep_row = {
            "epoch": ep,
            "time_sec": round(time.time() - start, 2),
            "train_acc": float(run_correct / max(1, run_total)),
            "train_loss_ema": float(run_loss / max(1, run_total)),
            "val_acc": float(val_acc),
            "val_f1_macro": float(val_f1),
            "val_loss": float(val_loss),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        logs.append(ep_row)
        if progress_path:
            with contextlib.suppress(Exception), open(progress_path, "a") as pf:
                pf.write(json.dumps(ep_row) + "\n")
        if task_ctx is not None:
            with contextlib.suppress(Exception):
                task_ctx.update_state(state="STARTED", meta={"progress_epoch": ep, "epochs_total": cfg.epochs, "last_val": {"acc": val_acc, "f1": val_f1}})

        if val_acc > best["val_acc"]:
            best = {"val_acc": float(val_acc), "epoch": ep, "val_f1": float(val_f1)}
            patience_counter = 0
            if save_dir and save_artifacts:
                with contextlib.suppress(Exception):
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
        else:
            patience_counter += 1

        if cfg.early_stopping_patience > 0 and patience_counter >= cfg.early_stopping_patience:
            logger.info("Early stopping at epoch %d (no improvement for %d epochs)", ep, cfg.early_stopping_patience)
            break

    return {
        "seed": cfg.seed,
        "device": str(device),
        "cfg": asdict(cfg),
        "best": best,
        "epochs": logs,
        "save_dir": save_dir if save_dir else None,
    }


__all__ = [
    "MLP",
    "AlcnetCfg",
    "AttnCNN",
    "ConvBlock",
    "DropPath",
    "FiLMGenerator",
    "GatedHybridClassifier",
    "ModelEMA",
    "SEBlock",
    "apply_film",
    "attn_stats_21",
    "build_attn_tensor",
    "pick_device",
    "set_all_seeds",
    "train_and_eval",
]

