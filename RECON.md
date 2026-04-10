# RECON.md — ALCNet Repository Reconnaissance

## Project Identity

**Name**: ALCNet — Attention-Layer Convolutional Network  
**Domain**: Deep Learning / NLU — a research project exploring whether convolutional processing of transformer attention maps can improve classification beyond vanilla fine-tuning.  
**License**: MIT  
**Author**: Srinivas Harish  
**Language**: Python 3 (PyTorch + HuggingFace ecosystem)  

## What It Does

ALCNet is a hybrid transformer-CNN architecture for natural language understanding. The core idea:

1. Run input through a pretrained transformer (RoBERTa-large)
2. Extract multi-head attention tensors from selected layers
3. Process those attention maps as 2D images through a CNN
4. Compute 21 handcrafted statistical features from the attention matrices
5. Fuse three branches (CLS token, CNN features, statistical features) via a learned confidence gate
6. The key insight: gradients flow back through the CNN into the transformer's Q/K/V, reshaping attention allocation

**Task**: GLUE RTE (Recognizing Textual Entailment) — binary classification (entailment vs not-entailment), 2490 training examples, 277 validation examples.

## Architecture Map

```
src/
  main.py     (550 LOC) — Core: model definitions, training loop, data loading, attention utilities
  api.py      (91 LOC)  — FastAPI REST API for submitting/querying training runs
  db.py       (241 LOC) — SQLAlchemy ORM (SQLite) for run tracking, epoch metrics, artifacts
  worker.py   (128 LOC) — Celery background worker that runs training tasks
  runner.py   (84 LOC)  — CLI runner + config builder from request dicts
  __init__.py (2 LOC)   — Empty

Makefile      — Dev startup targets (redis, api, worker, deps)
Procfile.dev  — Honcho process manager config (redis + uvicorn + celery)
```

### Module Dependency Graph
```
api.py ──> db.py
       ──> worker.py ──> runner.py ──> main.py
                     ──> db.py
```

### Model Architecture (GatedHybridClassifier)
```
Input: (premise, hypothesis) pair
  │
  ▼
EncoderWrapper (RoBERTa-large, output_attentions=True)
  │
  ├──> CLS token vector (1024-d) ──> Linear ──> vanilla_logits
  │
  ├──> Attention tensors [layers -3,-2,-1] ──> build_attn_tensor() ──> AttnCNN ──> 256-d
  │                                                                                  │
  ├──> Attention tensors ──> attn_stats_21() ──> 21 scalar features ──> MLP ──> 128-d │
  │                                                                                  │
  │                                                                    concat ◄──────┘
  │                                                                      │
  │                                                              cnn_head ──> cnn_logits
  │                                                              conf_head ──> confidence σ
  │
  ▼
  logits = (1 - conf) * vanilla_logits + conf * cnn_logits
```

### AttnCNN Architecture
```
Conv2d(in_ch, 64, 1x1) → BN → ReLU           (channel adapter)
Conv2d(64, 128, 3x3) → BN → ReLU → Drop2d    (ConvBlock)
MaxPool2d(2)
Conv2d(128, 256, 3x3) → BN → ReLU → Drop2d   (ConvBlock)
AdaptiveAvgPool2d(1)
Linear(256, 256)
```

### 21 Statistical Features (attn_stats_21)
Per selected layer (3 layers x 6 features = 18):
- Long-range attention ratio (k=5)
- CLS-in attention mean
- Entropy mean
- Entropy std across heads
- Max-row attention (normalized)
- Asymmetry score

Plus 3 global features from last selected layer:
- Diagonal mass
- Off-diagonal mass
- Effective heads ratio (off_mass > 0.60)

## External Dependencies

| Package | Version | Role |
|---------|---------|------|
| torch | 2.10.0+cu128 | Core DL framework |
| transformers | 5.2.0 | Pretrained models (RoBERTa-large) |
| datasets | 4.8.4 | GLUE data loading |
| scikit-learn | 1.7.2 | Metrics (accuracy, F1) |
| numpy | 2.2.6 | Numerical ops |
| tqdm | 4.67.1 | Progress bars |
| fastapi | 0.135.1 | REST API |
| uvicorn | 0.27.1 | ASGI server |
| celery | 5.3.6 | Task queue |
| sqlalchemy | ??? | ORM (imported but may not be installed) |
| pydantic | 2.12.5 | Request validation |
| redis | (system) | Celery broker |

**CRITICAL**: No requirements.txt, pyproject.toml, or setup.py exists.

## Database Schema

SQLite via SQLAlchemy ORM. Three tables:

- **runs**: id (UUID PK), task_id, status (QUEUED/RUNNING/COMPLETE/FAILED), ablation, overrides_json, epochs, batch_size, max_len, device, save_dir, save_artifacts, early_stop_json, best_val_acc, result_json, error, created_at, updated_at
- **epochs**: id (auto PK), run_id (FK), epoch, time_sec, train_acc, train_loss_ema, val_acc, val_f1_macro, lr, gates_json, created_at
- **artifacts**: id (auto PK), run_id (FK), kind, path, bytes, created_at

## API Surface

| Method | Path | Description |
|--------|------|-------------|
| GET | / | Health check |
| POST | /runs | Submit training run (enqueues Celery task) |
| GET | /runs | List runs (with limit, status filter) |
| GET | /runs/{run_id} | Get single run with epoch logs and artifacts |

## Critical Gaps Identified

### Code vs Documentation Mismatch
- **README describes FiLM conditioning** (Stats -> gamma/beta -> modulate CNN intermediate layers) but **code does NOT implement FiLM**. The stats MLP output is simply concatenated with CNN features — no multiplicative/additive conditioning.

### Infrastructure
- No requirements.txt / pyproject.toml
- No tests (0% coverage, tests/ in .gitignore)
- No CI/CD pipeline
- No pre-commit hooks
- No type checking configuration
- dump.rdb (Redis dump file) committed to repo
- env.txt is empty (should it be .env?)

### Code Quality
- 15 ruff violations (unused imports, multiple statements per line)
- Bare `except:` clause in db.py:103
- `datetime.utcnow()` deprecated (should use `datetime.now(UTC)`)
- Global mutable `_MASK_CACHE` dict — not thread-safe
- Silent exception swallowing throughout (try/except pass patterns)
- Hardcoded CORS origins
- No input validation on `overrides` dict in API (arbitrary setattr on config)

### ML/Research
- Reported accuracy: 52.71% — far below RoBERTa-large RTE baseline (~86-88%)
- Missing FiLM implementation despite README claiming it
- No learning rate warmup (uses CosineAnnealingLR from step 0)
- No early stopping
- No gradient accumulation for effective larger batch sizes
- attn_stats_21 runs under `@torch.no_grad()` but is computed during training — stats don't receive gradients even though they could inform the model
- Scheduler steps per batch instead of per epoch (unusual but possibly intentional)

### Performance
- No mixed precision for attention stat computation
- Attention tensors are fully materialized and interpolated every batch
- No caching of tokenized data between epochs
- DataLoader num_workers=0 when not in Celery

## SOTA Context

- **GLUE RTE SOTA**: ~92-95% accuracy (large models, multi-task learning)
- **RoBERTa-large baseline on RTE**: ~86-88% (standard fine-tuning)
- **This project**: 52.71% (early experimental, below random for 2-class)
- **Key competitors/related work**: Attention rollout methods, CFFormer (CNN-Transformer hybrid), HAM (Hybrid Attention Module), LoRA/Adapter-based fine-tuning
- **FiLM conditioning** (Perez et al. 2018): Well-established technique for conditional feature modulation — the README's described architecture is sound but unimplemented

## File Inventory

| File | Lines | Last Modified |
|------|-------|---------------|
| src/main.py | 550 | Initial scaffold |
| src/db.py | 241 | Initial scaffold |
| src/worker.py | 128 | Initial scaffold |
| src/api.py | 91 | Initial scaffold |
| src/runner.py | 84 | Initial scaffold |
| src/__init__.py | 2 | Initial scaffold |
| Makefile | 33 | Initial scaffold |
| Procfile.dev | 4 | Initial scaffold |
| README.md | 125 | Updated |
| .gitignore | 42 | Initial scaffold |
| LICENSE | 22 | Initial |
| env.txt | 1 (empty) | Initial |
| dump.rdb | binary | Should not be committed |
| **Total Python** | **1096** | |
