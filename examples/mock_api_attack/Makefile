.PHONY: install run-server attack all clean

install:
	uv sync

run-server:
	@echo "Starting Mock API Server..."
	uv run python server.py > server.log 2>&1 & echo $$! > server.pid
	@echo "Server started with PID $$(cat server.pid). Waiting for startup..."
	@sleep 10

attack:
	@echo "Running Attack..."
	uv run python attack.py

clean:
	@if [ -f server.pid ]; then \
		echo "Stopping server (PID $$(cat server.pid))..."; \
		kill $$(cat server.pid) || true; \
		rm server.pid; \
	fi
	@rm -f server.log

all: install run-server attack clean
