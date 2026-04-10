import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .db import (
    create_run,
    get_run_row,
    get_session,
    init_db,
    serialize_run,
    set_task_id,
)
from .db import (
    list_runs as db_list_runs,
)
from .worker import run_train_task

app = FastAPI(title="alcnet Service", version="0.1.0")

# CORS: configurable via CORS_ORIGINS env var (comma-separated), defaults to common local dev ports
_DEFAULT_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:5174", "http://127.0.0.1:5174",
    "http://localhost:5175", "http://127.0.0.1:5175",
]
_cors_origins_env = os.getenv("CORS_ORIGINS")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] if _cors_origins_env else _DEFAULT_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure tables exist on API startup
init_db()


class RunRequest(BaseModel):
    ablation: Literal["no_cnn", "no_stats", "no_film"] | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    epochs: int = Field(3, ge=1, le=100)
    batch_size: int = Field(8, ge=1, le=256)
    max_len: int = Field(128, ge=16, le=512)
    save_dir: str | None = None
    save_artifacts: bool = False
    gradient_checkpointing: bool = True
    device: str | None = None


@app.get("/")
def root():
    return {"ok": True, "msg": "alcnet service ready", "endpoints": ["/runs (POST)", "/runs (GET)", "/runs/{run_id} (GET)"]}


@app.post("/runs")
def submit_run(req: RunRequest):
    with get_session() as s:
        run = create_run(
            s,
            ablation=req.ablation,
            overrides=req.overrides,
            epochs=req.epochs,
            batch_size=req.batch_size,
            max_len=req.max_len,
            device=req.device or "auto",
            save_dir=req.save_dir,
            save_artifacts=req.save_artifacts,
            early_stop=None,
        )
        run_id = run.id

    # Enqueue Celery task
    async_result = run_train_task.delay(run_id, req.model_dump())

    with get_session() as s:
        set_task_id(s, run_id, async_result.id)

    return {"run_id": run_id, "task_id": async_result.id, "status": "QUEUED"}


@app.get("/runs")
def list_runs(limit: int = Query(50, ge=1, le=500), status: str | None = None):
    with get_session() as s:
        rows = db_list_runs(s, limit=limit, status=status)
        return [serialize_run(r, with_children=False) for r in rows]


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    with get_session() as s:
        row = get_run_row(s, run_id)
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        return serialize_run(row, with_children=True)

