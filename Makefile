.PHONY: help install data synthetic eda train test lint format api ui docker clean

help:
	@echo "install    Install Python + Node dependencies"
	@echo "data       Download the Kaggle dataset (needs credentials)"
	@echo "synthetic  Generate the labelled synthetic smoke fixture"
	@echo "train      Run the full training pipeline"
	@echo "test       Run all tests"
	@echo "lint       Lint Python and JavaScript"
	@echo "format     Auto-format Python"
	@echo "api        Run the FastAPI backend on :8000"
	@echo "ui         Run the React dev server on :5173"
	@echo "mlflow     Open the MLflow UI on :5000"
	@echo "docker     Build and start everything with Docker Compose"

install:
	pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"
	pip install -e "./ml[dev,boost,tracking,viz]"
	pip install -r backend/requirements.txt
	cd frontend && npm install

data:
	python ml/scripts/download_data.py

synthetic:
	python ml/scripts/make_synthetic_data.py

train:
	python ml/scripts/train.py

train-synthetic:
	python ml/scripts/train.py --synthetic

test:
	pytest

lint:
	ruff check ml/src ml/scripts ml/tests backend/app backend/tests
	cd frontend && npm run lint

format:
	ruff format ml/src ml/scripts backend/app

api:
	MODEL_DIR=models/fraud_detector uvicorn app.main:app --app-dir backend --reload --port 8000

ui:
	cd frontend && npm run dev

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker:
	docker compose up --build

clean:
	rm -rf reports/*.csv reports/*.json reports/figures/*.png
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
