# Knowledge Graph Reasoning with R-GCN

A knowledge-graph link-prediction system built on a **Relational Graph Convolutional Network (R-GCN)**. Graph data is stored in **Neo4j**, models are trained in PyTorch with Weights & Biases tracking, and predictions are served through a **FastAPI** service with a **Streamlit** UI.

## What's implemented

| Component | File(s) | Details |
|---|---|---|
| Data ingestion | `src/etl/fb15k_ingestion.py`, `src/etl/neo4j_ingestion.py`, `scripts/data_ingestion.py` | Loads the **FB15k-237** benchmark (or a small sample graph) into Neo4j |
| R-GCN model | `src/models/rgcn.py` | R-GCN encoder for multi-relational link prediction, plus a variant with a contrastive-learning objective |
| Training | `src/train/trainer.py`, `scripts/train_model.py` | Training loop with validation (MRR-based model selection), LR scheduling, checkpointing, and W&B logging |
| Trained checkpoint | `models/r-gcn_1760253457/` | Saved R-GCN model weights |
| API | `src/api/` | FastAPI routes for link prediction (`/link`), similar entities, entity relations, subgraph lookup, graph stats, and reasoning paths |
| UI | `src/ui/streamlit_app.py` | Streamlit app to query predictions and explore the graph |
| Infrastructure | `infra/docker/` | Docker Compose with Neo4j, Redis, the API, the UI, MLflow, Postgres, Prometheus, and Grafana |

## Current status and roadmap
- ✅ FB15k-237 ingestion into Neo4j
- ✅ R-GCN training and a saved checkpoint
- ✅ Link-prediction API and Streamlit UI
- ⏳ Graph Transformer model (option exists in the training script; model not yet implemented)
- ⏳ Explainability module and Airflow/Prefect pipeline orchestration (folders scaffolded)
- ⏳ Broader test coverage (currently basic config, model, and data tests)

## Project structure
```
config/            app and model config
scripts/           data ingestion, training, serving, checkpoint utilities
src/etl/           Neo4j and FB15k-237 ingestion
src/models/        R-GCN model
src/train/         training loop
src/api/           FastAPI app and routers
src/ui/            Streamlit app
infra/             Docker and monitoring config
tests/             basic tests
```

## Getting started
```bash
git clone https://github.com/TRasagna/Graph-Reasoning-AI-System-with-MLOps.git
cd Graph-Reasoning-AI-System-with-MLOps
make install
cp .env.example .env        # set Neo4j credentials

# Start Neo4j and supporting services
make deploy

# Load data (sample, fb15k-237, or wn18rr)
python scripts/data_ingestion.py --dataset fb15k-237

# Train R-GCN
python scripts/train_model.py --architecture r-gcn --epochs 100

# Serve the API and UI
make serve
make ui
```

## Tech stack
Python · PyTorch · Graph Neural Networks (R-GCN) · Neo4j · FastAPI · Streamlit · Weights & Biases · Docker · Prometheus · Grafana
