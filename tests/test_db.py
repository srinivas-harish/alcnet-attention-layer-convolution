"""DB model and helper tests."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db import (
    Base,
    Run,
    add_artifact,
    append_epoch,
    complete_run,
    create_run,
    list_runs,
    serialize_run,
    set_task_id,
    update_status,
)


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database and session for testing."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    session = Session()
    yield session
    session.close()


class TestCreateRun:
    def test_creates_run_with_defaults(self, db_session):
        run = create_run(
            db_session,
            ablation=None,
            overrides=None,
            epochs=3,
            batch_size=8,
            max_len=128,
            device="cpu",
            save_dir=None,
            save_artifacts=False,
            early_stop=None,
        )
        db_session.commit()
        assert run.id is not None
        assert run.status == "QUEUED"
        assert run.epochs == 3

    def test_creates_run_with_ablation(self, db_session):
        run = create_run(
            db_session,
            ablation="no_cnn",
            overrides={"lr_head": 1e-3},
            epochs=5,
            batch_size=16,
            max_len=256,
            device="cuda",
            save_dir="/tmp/test",
            save_artifacts=True,
            early_stop={"patience": 3},
        )
        db_session.commit()
        assert run.ablation == "no_cnn"
        assert run.overrides_json == {"lr_head": 1e-3}
        assert run.save_artifacts is True


class TestSetTaskId:
    def test_sets_task_id(self, db_session):
        run = create_run(
            db_session, ablation=None, overrides=None, epochs=3,
            batch_size=8, max_len=128, device="cpu",
            save_dir=None, save_artifacts=False, early_stop=None,
        )
        db_session.commit()
        set_task_id(db_session, run.id, "celery-task-123")
        db_session.commit()
        refreshed = db_session.get(Run, run.id)
        assert refreshed.task_id == "celery-task-123"

    def test_nonexistent_run_does_not_crash(self, db_session):
        set_task_id(db_session, "nonexistent-id", "task-123")


class TestUpdateStatus:
    def test_updates_status(self, db_session):
        run = create_run(
            db_session, ablation=None, overrides=None, epochs=3,
            batch_size=8, max_len=128, device="cpu",
            save_dir=None, save_artifacts=False, early_stop=None,
        )
        db_session.commit()
        update_status(db_session, run.id, "RUNNING")
        db_session.commit()
        refreshed = db_session.get(Run, run.id)
        assert refreshed.status == "RUNNING"

    def test_updates_status_with_error(self, db_session):
        run = create_run(
            db_session, ablation=None, overrides=None, epochs=3,
            batch_size=8, max_len=128, device="cpu",
            save_dir=None, save_artifacts=False, early_stop=None,
        )
        db_session.commit()
        update_status(db_session, run.id, "FAILED", error="OOM")
        db_session.commit()
        refreshed = db_session.get(Run, run.id)
        assert refreshed.status == "FAILED"
        assert refreshed.error == "OOM"


class TestAppendEpoch:
    def test_appends_epoch(self, db_session):
        run = create_run(
            db_session, ablation=None, overrides=None, epochs=3,
            batch_size=8, max_len=128, device="cpu",
            save_dir=None, save_artifacts=False, early_stop=None,
        )
        db_session.commit()
        append_epoch(db_session, run.id, {
            "epoch": 1, "time_sec": 10.5, "train_acc": 0.6,
            "train_loss_ema": 0.5, "val_acc": 0.55, "val_f1_macro": 0.54,
            "lr": 1e-4, "gates": None,
        })
        db_session.commit()
        refreshed = db_session.get(Run, run.id)
        assert len(refreshed.epochs_rel) == 1
        assert refreshed.epochs_rel[0].epoch == 1


class TestCompleteRun:
    def test_completes_run(self, db_session):
        run = create_run(
            db_session, ablation=None, overrides=None, epochs=3,
            batch_size=8, max_len=128, device="cpu",
            save_dir=None, save_artifacts=False, early_stop=None,
        )
        db_session.commit()
        complete_run(db_session, run.id, result_json={"best": {"val_acc": 0.85}}, best_val_acc=0.85)
        db_session.commit()
        refreshed = db_session.get(Run, run.id)
        assert refreshed.status == "COMPLETE"
        assert refreshed.best_val_acc == 0.85


class TestListRuns:
    def test_lists_runs(self, db_session):
        for _ in range(3):
            create_run(
                db_session, ablation=None, overrides=None, epochs=3,
                batch_size=8, max_len=128, device="cpu",
                save_dir=None, save_artifacts=False, early_stop=None,
            )
        db_session.commit()
        runs = list_runs(db_session, limit=10)
        assert len(list(runs)) == 3

    def test_limits_results(self, db_session):
        for _ in range(5):
            create_run(
                db_session, ablation=None, overrides=None, epochs=3,
                batch_size=8, max_len=128, device="cpu",
                save_dir=None, save_artifacts=False, early_stop=None,
            )
        db_session.commit()
        runs = list_runs(db_session, limit=2)
        assert len(list(runs)) == 2

    def test_filters_by_status(self, db_session):
        run1 = create_run(
            db_session, ablation=None, overrides=None, epochs=3,
            batch_size=8, max_len=128, device="cpu",
            save_dir=None, save_artifacts=False, early_stop=None,
        )
        create_run(
            db_session, ablation=None, overrides=None, epochs=3,
            batch_size=8, max_len=128, device="cpu",
            save_dir=None, save_artifacts=False, early_stop=None,
        )
        db_session.commit()
        update_status(db_session, run1.id, "COMPLETE")
        db_session.commit()
        complete_runs = list_runs(db_session, status="COMPLETE")
        assert len(list(complete_runs)) == 1


class TestSerializeRun:
    def test_serialize_without_children(self, db_session):
        run = create_run(
            db_session, ablation="test", overrides={"lr_head": 1e-3},
            epochs=5, batch_size=16, max_len=128, device="cpu",
            save_dir=None, save_artifacts=False, early_stop=None,
        )
        db_session.commit()
        d = serialize_run(run, with_children=False)
        assert d["run_id"] == run.id
        assert d["ablation"] == "test"
        assert d["epochs"] == 5
        assert "epochs_log" not in d

    def test_serialize_with_children(self, db_session):
        run = create_run(
            db_session, ablation=None, overrides=None, epochs=3,
            batch_size=8, max_len=128, device="cpu",
            save_dir=None, save_artifacts=False, early_stop=None,
        )
        db_session.commit()
        append_epoch(db_session, run.id, {
            "epoch": 1, "time_sec": 10.5, "train_acc": 0.6,
            "train_loss_ema": 0.5, "val_acc": 0.55, "val_f1_macro": 0.54,
            "lr": 1e-4, "gates": None,
        })
        add_artifact(db_session, run.id, "report", "/tmp/report.json", bytes=1024)
        db_session.commit()
        # Refresh to get relationships
        refreshed = db_session.get(Run, run.id)
        d = serialize_run(refreshed, with_children=True)
        assert "epochs_log" in d
        assert len(d["epochs_log"]) == 1
        assert "artifacts" in d
        assert len(d["artifacts"]) == 1
        assert d["artifacts"][0]["kind"] == "report"
