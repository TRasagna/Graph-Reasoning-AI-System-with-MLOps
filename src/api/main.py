"""
FastAPI main application for the KG Reasoning System.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import logging

# Import local modules
from src.utils.config import Config, get_config
from src.utils.database import Neo4jConnection
from src.utils.logging import setup_logging, get_logger
from src.models.rgcn import RGCNModel
from src.api.routers import predict, graph, explain

# Setup logging
setup_logging()
logger = get_logger('api')

# Global variables for models and database
MODEL = None
DATABASE = None
CONFIG = None
MODEL_PATH_ATTEMPTED = None
MODEL_LOAD_ERROR = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global MODEL, DATABASE, CONFIG

    logger.info("Starting KG Reasoning API")

    try:
        # Load configuration
        CONFIG = get_config()

        # Connect to database
        DATABASE = Neo4jConnection(
            uri=CONFIG.database.neo4j_uri,
            username=CONFIG.database.neo4j_username,
            password=CONFIG.database.neo4j_password,
            database=CONFIG.database.neo4j_database
        )

        # Load model if available. Support models saved in timestamped subdirectories
        models_dir = Path(CONFIG.paths.models_dir)
        model_path = None

        if models_dir.exists():
            # Prefer explicit best_model.pt, then final_model.pt, otherwise pick newest .pt
            best_candidates = list(models_dir.rglob('best_model.pt'))
            final_candidates = list(models_dir.rglob('final_model.pt'))
            all_candidates = best_candidates + final_candidates + list(models_dir.rglob('*.pt'))

            if all_candidates:
                # pick the most recently modified candidate
                model_path = max(all_candidates, key=lambda p: p.stat().st_mtime)

        if model_path and model_path.exists():
            try:
                MODEL, _ = RGCNModel.load_model(model_path)
                MODEL.eval()
                MODEL_PATH_ATTEMPTED = str(model_path)
                MODEL_LOAD_ERROR = None
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                MODEL_PATH_ATTEMPTED = str(model_path)
                MODEL_LOAD_ERROR = str(e)
                logger.warning(f"Failed to load model from {model_path}: {e}")
        else:
            MODEL_PATH_ATTEMPTED = str(models_dir)
            MODEL_LOAD_ERROR = "no_checkpoint_found"
            logger.warning(f"No model checkpoint found under {models_dir}")

        # Expose objects to app.state for dependency injection
        try:
            app.state.model = MODEL
            app.state.database = DATABASE
            app.state.config = CONFIG
        except Exception:
            pass

        logger.info("KG Reasoning API started successfully")

    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down KG Reasoning API")
    if DATABASE:
        DATABASE.close()

    # Clear app.state entries
    try:
        app.state.model = None
        app.state.database = None
        app.state.config = None
    except Exception:
        pass


# Create FastAPI app
app = FastAPI(
    title="KG Reasoning API",
    description="Graph Neural Network-based Knowledge Graph Reasoning System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency functions
def get_model():
    """Get the loaded model."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return MODEL

def get_database():
    """Get the database connection."""
    if DATABASE is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return DATABASE

def get_app_config():
    """Get the application configuration."""
    if CONFIG is None:
        raise HTTPException(status_code=503, detail="Configuration not available")
    return CONFIG

# Include routers
app.include_router(predict.router, prefix="/api/v1/predict", tags=["prediction"])
app.include_router(graph.router, prefix="/api/v1/graph", tags=["graph"])
app.include_router(explain.router, prefix="/api/v1/explain", tags=["explanation"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    status = {
        "status": "healthy",
        "service": "kg-reasoning-api",
        "version": "1.0.0",
        "model_loaded": MODEL is not None,
        "database_connected": DATABASE is not None
    }

    # Add model diagnostics
    try:
        status['model_path_attempted'] = MODEL_PATH_ATTEMPTED
        status['model_load_error'] = MODEL_LOAD_ERROR
    except Exception:
        pass

    if DATABASE:
        try:
            stats = DATABASE.get_graph_statistics()
            status["graph_stats"] = stats
        except Exception as e:
            status["database_error"] = str(e)

    return status


@app.get("/debug/routes")
async def list_routes():
    """Return a list of registered routes (path and methods) for debugging."""
    routes = []
    for r in app.routes:
        try:
            methods = list(r.methods) if hasattr(r, 'methods') and r.methods else []
            routes.append({
                'path': getattr(r, 'path', str(r)),
                'name': getattr(r, 'name', None),
                'methods': methods
            })
        except Exception:
            continue
    return {'routes': routes}


@app.get("/debug/mappings")
async def debug_mappings():
    """Return a small diagnostic snapshot of database entity/relation mappings."""
    try:
        db = getattr(app.state, 'database', None)
        if db is None:
            return {"error": "database not available"}

        # Try to access mapping attributes
        etoi = getattr(db, '_entity_to_id', None)
        itoe = getattr(db, '_id_to_entity', None)
        rtoi = getattr(db, '_relation_to_id', None)
        itor = getattr(db, '_id_to_relation', None)

        def sample_keys(d, n=50):
            if not d:
                return []
            keys = list(d.keys())
            return keys[:n]

        return {
            'num_entities': len(etoi) if etoi is not None else 0,
            'sample_entities': sample_keys(etoi),
            'num_relations': len(rtoi) if rtoi is not None else 0,
            'sample_relations': sample_keys(rtoi),
            'db_build_mappings_info': {
                'entity_map_present': etoi is not None,
                'relation_map_present': rtoi is not None
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/info")
async def get_system_info():
    """Get system information."""
    if not DATABASE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        graph_stats = DATABASE.get_graph_statistics()

        return {
            "system": {
                "name": "KG Reasoning System",
                "version": "1.0.0",
                "model_architecture": CONFIG.model.architecture if CONFIG else "unknown"
            },
            "graph_statistics": graph_stats,
            "endpoints": {
                "prediction": "/api/v1/predict",
                "graph": "/api/v1/graph", 
                "explanation": "/api/v1/explain"
            }
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system information")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


def main():
    """Main entry point for running the server."""
    config = get_config()

    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
