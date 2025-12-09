.PHONY: install attack all

install:
	uv sync

attack:
	@echo "Running Attack..."
	uv run python attack.py

all: install attack
