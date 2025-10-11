# Makefile for KG Reasoner Project

.PHONY: help install setup-dev test clean build deploy data train serve

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  setup-dev    Setup development environment"
	@echo "  test         Run tests"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build Docker images"
	@echo "  deploy       Deploy with Docker Compose"
	@echo "  data         Setup sample data"
	@echo "  train        Train models"
	@echo "  serve        Start API server"
	@echo "  ui           Start Streamlit UI"

# Installation
install:
	pip install -r infra/docker/requirements.txt

setup-dev:
	pip install -r infra/docker/requirements.txt
	pip install pytest pytest-asyncio pytest-cov black isort flake8 mypy
	@echo "Development environment setup complete"

# Testing
test:
	pytest tests/ -v --cov=src

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/

# Docker operations
build:
	docker build -f infra/docker/Dockerfile -t kg-reasoner:latest .

deploy:
	docker-compose -f infra/docker/docker-compose.yml up -d

stop:
	docker-compose -f infra/docker/docker-compose.yml down

logs:
	docker-compose -f infra/docker/docker-compose.yml logs -f

# Data operations
data:
	python scripts/data_ingestion.py --dataset sample

data-fb15k:
	python scripts/data_ingestion.py --dataset fb15k-237

# Model training
train:
	python scripts/train_model.py --architecture r-gcn --epochs 50

train-transformer:
	python scripts/train_model.py --architecture graph-transformer --epochs 50

# Services
serve:
	python scripts/serve.py

ui:
	streamlit run src/ui/streamlit_app.py

# Development workflow
dev-setup: install data deploy
	@echo "Development environment ready!"
	@echo "API: http://localhost:8000"
	@echo "UI: http://localhost:8501"
	@echo "Neo4j: http://localhost:7474"

# Health checks
health:
	curl -f http://localhost:8000/health || exit 1

# Format code
format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

# Run linting
lint:
	flake8 src/ tests/ scripts/
	mypy src/

.DEFAULT_GOAL := help
