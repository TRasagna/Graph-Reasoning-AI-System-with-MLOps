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
from ..utils.config import Config, get_config
from ..utils.database import Neo4jConnection
from ..utils.logging import setup_logging, get_logger
from ..models.rgcn import RGCNModel
from .routers import predict, graph, explain

# Setup logging
setup_logging()
logger = get_logger('api')

# Global variables for models and database
MODEL = None
DATABASE = None
CONFIG = None


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

        # Load model if available
        model_path = Path(CONFIG.paths.models_dir) / "best_model.pt"
        if model_path.exists():
            try:
                MODEL, _ = RGCNModel.load_model(model_path)
                MODEL.eval()
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
        else:
            logger.warning(f"Model not found at {model_path}")

        logger.info("KG Reasoning API started successfully")

    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down KG Reasoning API")
    if DATABASE:
        DATABASE.close()


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
app.include_router(
    predict.router,
    prefix="/api/v1/predict",
    tags=["prediction"],
    dependencies=[Depends(get_model), Depends(get_database)]
)

app.include_router(
    graph.router,
    prefix="/api/v1/graph",
    tags=["graph"],
    dependencies=[Depends(get_database)]
)

app.include_router(
    explain.router,
    prefix="/api/v1/explain",
    tags=["explanation"],
    dependencies=[Depends(get_model), Depends(get_database)]
)


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

    if DATABASE:
        try:
            stats = DATABASE.get_graph_statistics()
            status["graph_stats"] = stats
        except Exception as e:
            status["database_error"] = str(e)

    return status


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
