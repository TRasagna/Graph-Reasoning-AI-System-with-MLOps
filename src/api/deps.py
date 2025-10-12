from fastapi import Request, HTTPException


def get_model(request: Request):
    """Dependency to retrieve the loaded model from app state."""
    model = getattr(request.app.state, 'model', None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return model


def get_database(request: Request):
    """Dependency to retrieve the database connection from app state."""
    db = getattr(request.app.state, 'database', None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db


def get_app_config(request: Request):
    cfg = getattr(request.app.state, 'config', None)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Configuration not available")
    return cfg
