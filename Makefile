    .PHONY: setup lint format test api

    setup:
	pip install -U pip
	pip install -e .[dev,test]
	pre-commit install

    lint:
	ruff check

    format:
	ruff check --fix
	ruff format

    test:
	pytest -q

    api:
	uvicorn neuralcache.api.server:app --reload --port 8080
