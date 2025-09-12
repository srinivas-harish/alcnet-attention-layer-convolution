SHELL := /bin/bash

# Ports can be overridden: e.g. REDIS_PORT=6380 API_PORT=8010 make up
REDIS_PORT ?= 6379
API_PORT ?= 8000

.PHONY: up
up:
	@echo "Starting redis:${REDIS_PORT}, api:${API_PORT}, worker via honcho..."
	@honcho -f Procfile.dev start

.PHONY: up-detach
up-detach:
	@echo "Starting in background via honcho (tmux-like, not daemonized)."
	@honcho -f Procfile.dev start &

.PHONY: api
api:
	REDIS_URL=redis://localhost:$(REDIS_PORT)/0 uvicorn src.api:app --host 0.0.0.0 --port $(API_PORT) --reload

.PHONY: worker
worker:
	REDIS_URL=redis://localhost:$(REDIS_PORT)/0 celery -A src.worker.celery_app worker --loglevel=INFO --pool=solo

.PHONY: redis
redis:
	redis-server --port $(REDIS_PORT)

.PHONY: deps
deps:
	python -m pip install --upgrade pip
	pip install honcho fastapi uvicorn celery[redis] redis sqlalchemy pydantic
