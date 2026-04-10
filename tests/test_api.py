"""Tests for the FastAPI application endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.db import Base


@pytest.fixture
def test_db():
    """Create an in-memory SQLite database for API testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    return TestSession


@pytest.fixture
def client(test_db):
    """Create a test client with mocked database and Celery."""
    # Patch get_session to use test DB
    from contextlib import contextmanager

    @contextmanager
    def mock_get_session():
        s = test_db()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    with (
        patch("src.api.get_session", new=mock_get_session),
        patch("src.api.init_db"),
        patch("src.api.run_train_task") as mock_task,
    ):
        mock_result = MagicMock()
        mock_result.id = "fake-celery-task-id"
        mock_task.delay.return_value = mock_result

        from src.api import app
        with TestClient(app) as tc:
            yield tc


class TestRootEndpoint:
    def test_root_returns_ok(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "endpoints" in data


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "torch_version" in data
        assert "cuda_available" in data


class TestSubmitRun:
    def test_submit_minimal_run(self, client):
        resp = client.post("/runs", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["status"] == "QUEUED"
        assert data["task_id"] == "fake-celery-task-id"

    def test_submit_run_with_overrides(self, client):
        resp = client.post("/runs", json={
            "epochs": 5,
            "batch_size": 16,
            "ablation": "no_cnn",
            "overrides": {"lr_head": 1e-3},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data

    def test_submit_invalid_ablation(self, client):
        resp = client.post("/runs", json={"ablation": "invalid_mode"})
        assert resp.status_code == 422

    def test_submit_invalid_epochs(self, client):
        resp = client.post("/runs", json={"epochs": 0})
        assert resp.status_code == 422

    def test_submit_run_returns_unique_ids(self, client):
        resp1 = client.post("/runs", json={})
        resp2 = client.post("/runs", json={})
        assert resp1.json()["run_id"] != resp2.json()["run_id"]


class TestListRuns:
    def test_list_empty(self, client):
        resp = client.get("/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_after_submit(self, client):
        client.post("/runs", json={"epochs": 1})
        client.post("/runs", json={"epochs": 2})
        resp = client.get("/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) == 2

    def test_list_with_limit(self, client):
        for _ in range(5):
            client.post("/runs", json={})
        resp = client.get("/runs?limit=3")
        assert resp.status_code == 200
        assert len(resp.json()) == 3


class TestGetRun:
    def test_get_existing_run(self, client):
        submit_resp = client.post("/runs", json={"epochs": 3, "batch_size": 16})
        run_id = submit_resp.json()["run_id"]

        resp = client.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert data["epochs"] == 3
        assert data["batch_size"] == 16
        assert "epochs_log" in data
        assert "artifacts" in data

    def test_get_nonexistent_run(self, client):
        resp = client.get("/runs/nonexistent-id")
        assert resp.status_code == 404
