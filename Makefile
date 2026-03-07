PYTHON := .venv/bin/python
APP := dashboardv2.py
PORT := 8501
HOST := 127.0.0.1

.PHONY: run status stop

run:
	$(PYTHON) -m streamlit run $(APP) --server.port $(PORT) --server.address $(HOST)

status:
	@lsof -nP -iTCP:$(PORT) -sTCP:LISTEN || echo "No app listening on port $(PORT)"

stop:
	@pids=$$(lsof -tiTCP:$(PORT) -sTCP:LISTEN); \
	if [ -n "$$pids" ]; then \
		echo "Stopping PID(s): $$pids"; \
		kill $$pids; \
	else \
		echo "No app listening on port $(PORT)"; \
	fi
