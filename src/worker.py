# Celery worker for alcnet
import os
import json
import threading
import time
import shutil
from typing import Optional

from celery import Celery

from .main import AlcnetCfg
from .db import (
    init_db, get_session, update_status, append_epoch, complete_run, add_artifact, get_run_row
)
from .runner import run_from_req


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("alcnet_service", broker=REDIS_URL, backend=REDIS_URL)

# Mark Celery execution for DataLoader safety
os.environ["IN_CELERY"] = "1"

# Ensure DB is ready in worker process
init_db()


@celery_app.task(name="worker.run_train_task", bind=True)
def run_train_task(self, run_id: str, req: dict):
    """
    Background training task. Expects the exact POST body from the API.
    Writes per-epoch progress into DB and registers artifacts.
    """
    try:
        # Helper to update DB with retry (SQLite can be briefly locked)
        def db_retry(op, *a, **kw):
            for i in range(12):
                try:
                    with get_session() as s:
                        op(s, *a, **kw)
                    return True
                except Exception as e:
                    msg = str(e).lower()
                    if "locked" in msg or "busy" in msg:
                        time.sleep(0.05 * (i + 1))
                        continue
                    raise
            return False

        db_retry(lambda s: update_status(s, run_id, "RUNNING"))

        # Decide save_dir (default under repo src/runs/<run_id>)
        base_dir = os.path.dirname(__file__)
        save_dir = req.get("save_dir") or os.path.join(base_dir, "runs", run_id)
        os.makedirs(save_dir, exist_ok=True)

        # Persist the exact request used
        req_path = os.path.join(save_dir, "run_req.json")
        with open(req_path, "w") as f:
            json.dump(req, f, indent=2)

        # Progress path for per-epoch streaming
        progress_path = os.path.join(save_dir, "progress.jsonl")

        # Tailer thread to stream epochs into DB
        stop_evt = threading.Event()

        def tail_progress(p: str):
            pos = 0
            while not stop_evt.is_set():
                try:
                    if not os.path.exists(p):
                        time.sleep(0.3)
                        continue
                    with open(p, "r") as f:
                        f.seek(pos)
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                e = json.loads(line)
                            except Exception:
                                continue
                            db_retry(lambda s: append_epoch(s, run_id, e))
                        pos = f.tell()
                except Exception:
                    pass
                time.sleep(0.5)

        t = threading.Thread(target=tail_progress, args=(progress_path,), daemon=True)
        t.start()

        # Run training (in-process)
        result = run_from_req(req, run_id=run_id, save_dir=save_dir, progress_file=progress_path)

        stop_evt.set(); t.join(timeout=2.0)

        # Register obvious artifacts if present
        report_path = os.path.join(save_dir, "report.json")
        if os.path.exists(report_path):
            try:
                db_retry(lambda s: add_artifact(s, run_id, "report", report_path, bytes=os.path.getsize(report_path)))
            except Exception:
                pass
        model_path = os.path.join(save_dir, "model.pt")
        if os.path.exists(model_path):
            try:
                db_retry(lambda s: add_artifact(s, run_id, "checkpoint", model_path, bytes=os.path.getsize(model_path)))
            except Exception:
                pass

        # Mark complete with result
        db_retry(lambda s: complete_run(s, run_id, result_json=result, best_val_acc=(result.get("best") or {}).get("val_acc")))
        return {"ok": True, "run_id": run_id, "result": result}

    except Exception as e:
        try:
            with get_session() as s:
                from .db import get_run_row
                row = get_run_row(s, run_id)
                if row:
                    row.status = "FAILED"
                    row.error = str(e)
        except Exception:
            pass
        raise

