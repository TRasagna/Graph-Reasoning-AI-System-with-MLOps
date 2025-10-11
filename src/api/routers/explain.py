"""
Explainability endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from ...utils.logging import get_logger

logger = get_logger('api.explain')
router = APIRouter()


@router.post("/reasoning-paths")
async def get_reasoning_paths(
    head_entity: str,
    relation: str,
    max_length: int = 3,
    top_k: int = 5,
    model=Depends(lambda: None),
    db=Depends(lambda: None)
):
    """Get reasoning paths for a prediction."""
    try:
        head_id = db.get_entity_id(head_entity)
        rel_id = db.get_relation_id(relation)

        if head_id is None or rel_id is None:
            raise HTTPException(status_code=404, detail="Entity or relation not found")

        edge_index, edge_type = db.get_graph_data()

        if hasattr(model, 'get_reasoning_paths'):
            paths = model.get_reasoning_paths(
                head_id, rel_id, edge_index, edge_type, 
                max_path_length=max_length, top_k=top_k
            )
            return paths
        else:
            return {"message": "Reasoning paths not available for this model"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting reasoning paths: {e}")
        raise HTTPException(status_code=500, detail=str(e))
