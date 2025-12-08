PHONY: sync lock format

sync:
	uv sync

lock:
	uv lock

format: sync lock
	uv run black .
	uv run isort .
	uv run mypy .
