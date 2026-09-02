.PHONY: install dev infra up test lint eval clean

install:
	python -m pip install -e '.[dev]'

dev:
	uvicorn app.main:app --reload

infra:
	docker compose up -d neo4j qdrant

up:
	docker compose up --build

test:
	pytest

lint:
	ruff check app tests

eval:
	python -m app.eval_runner

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +
