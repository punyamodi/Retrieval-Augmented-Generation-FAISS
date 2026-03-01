.PHONY: install dev-install run-api run-ui run test lint format clean

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio ruff

run-api:
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run ui/Home.py --server.port 8501

run:
	@echo "Starting RAG Studio API and UI..."
	@start /B uvicorn app.api.main:app --host 0.0.0.0 --port 8000
	@streamlit run ui/Home.py --server.port 8501

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=app --cov-report=term-missing

lint:
	ruff check app/ tests/

format:
	ruff format app/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

setup-storage:
	mkdir -p storage/documents storage/indexes storage/sessions
