"""
Graph visualization and query endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

from ...utils.logging import get_logger

logger = get_logger('api.graph')
router = APIRouter()


@router.get("/subgraph/{entity_name}")
async def get_subgraph(
    entity_name: str,
    max_hops: int = 2,
    max_nodes: int = 50,
    db=Depends(lambda: None)
):
    """Get subgraph around an entity."""
    try:
        if not db.get_entity_id(entity_name):
            raise HTTPException(
                status_code=404,
                detail=f"Entity '{entity_name}' not found"
            )

        subgraph = db.query_subgraph(entity_name, max_hops, max_nodes)

        return {
            "center_entity": entity_name,
            "subgraph": subgraph,
            "num_nodes": len(subgraph.get('nodes', [])),
            "num_edges": len(subgraph.get('relationships', []))
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting subgraph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_graph_stats(db=Depends(lambda: None)):
    """Get graph statistics."""
    try:
        stats = db.get_graph_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
