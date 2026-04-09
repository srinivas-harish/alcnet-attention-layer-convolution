import argparse
import json
import logging
import os
from dataclasses import fields as dataclass_fields
from typing import Any

from .main import AlcnetCfg, pick_device, train_and_eval

logger = logging.getLogger(__name__)

# Whitelist of config fields that can be set via API overrides
_ALLOWED_OVERRIDE_KEYS = frozenset(f.name for f in dataclass_fields(AlcnetCfg))


def run_from_req(req: dict[str, Any], *, run_id: str | None = None, save_dir: str | None = None, progress_file: str | None = None) -> dict[str, Any]:
    # Device selection
    device = pick_device(req.get("device"))

    # Base config
    cfg = AlcnetCfg()

    # Direct training knobs
    if "epochs" in req:
        cfg.epochs = int(req["epochs"]) or cfg.epochs
    if "batch_size" in req:
        cfg.batch_size = int(req["batch_size"]) or cfg.batch_size
    if "max_len" in req:
        cfg.max_len = int(req["max_len"]) or cfg.max_len
    if "gradient_checkpointing" in req:
        cfg.gradient_checkpointing = bool(req["gradient_checkpointing"])
    if req.get("ablation"):
        cfg.ablation = str(req["ablation"])

    # Apply overrides into cfg — only allow known AlcnetCfg fields
    overrides = req.get("overrides") or {}
    for k, v in overrides.items():
        if k in _ALLOWED_OVERRIDE_KEYS:
            setattr(cfg, k, v)
        else:
            logger.warning("Ignoring unknown override key: %s", k)

    # Decide save_dir
    eff_save_dir = save_dir or req.get("save_dir")
    if eff_save_dir:
        os.makedirs(eff_save_dir, exist_ok=True)

    # Ensure at least one CUDA op runs to get profiling warmed up
    try:
        import torch
        if device.type == "cuda" and torch.cuda.is_available():
            _a = torch.randn(32, 32, device="cuda")
            _b = torch.randn(32, 32, device="cuda")
            _ = _a @ _b
            torch.cuda.synchronize()
    except Exception:
        pass

    result = train_and_eval(
        task_ctx=None,
        device=device,
        cfg=cfg,
        save_dir=eff_save_dir,
        save_artifacts=bool(req.get("save_artifacts", False)),
        progress_path=progress_file,
    )

    if eff_save_dir:
        out_path = os.path.join(eff_save_dir, "report.json")
        try:
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass

    return result


def main():
    p = argparse.ArgumentParser(description="Runner for alcnet training")
    p.add_argument("--infile", required=True, help="Path to JSON request payload (same shape as API body)")
    p.add_argument("--run-id", required=False, help="Run ID (for directory naming only)")
    p.add_argument("--save-dir", required=False, help="Override save_dir in payload")
    p.add_argument("--progress-file", required=False, help="If set, write per-epoch JSON lines for live progress")
    args = p.parse_args()

    with open(args.infile) as f:
        req = json.load(f)

    run_from_req(req, run_id=args.run_id, save_dir=args.save_dir, progress_file=args.progress_file)


if __name__ == "__main__":
    main()

