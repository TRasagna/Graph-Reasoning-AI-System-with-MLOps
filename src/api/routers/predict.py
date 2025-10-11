"""
Prediction endpoints for link prediction and entity reasoning.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import logging
import time

from ...utils.logging import get_logger

logger = get_logger('api.predict')
router = APIRouter()


class LinkPredictionRequest(BaseModel):
    """Request model for link prediction."""
    head_entity: str = Field(..., description="Head entity identifier")
    relation: str = Field(..., description="Relation type")
    tail_entity: Optional[str] = Field(None, description="Tail entity (optional)")
    top_k: int = Field(10, description="Number of top predictions to return")
    include_reasoning: bool = Field(False, description="Include reasoning paths")


class PredictionResult(BaseModel):
    """Single prediction result."""
    entity: str
    entity_id: int
    score: float
    rank: int


class LinkPredictionResponse(BaseModel):
    """Response model for link prediction."""
    head_entity: str
    relation: str
    predictions: List[PredictionResult]
    reasoning_paths: Optional[List[Dict[str, Any]]] = None
    model_confidence: float
    processing_time_ms: float


@router.post("/link", response_model=LinkPredictionResponse)
async def predict_link(
    request: LinkPredictionRequest,
    model=Depends(lambda: None),  # Will be injected by main app
    db=Depends(lambda: None)      # Will be injected by main app  
):
    """Predict missing links in the knowledge graph."""
    try:
        start_time = time.time()

        # Get entity and relation IDs
        head_id = db.get_entity_id(request.head_entity)
        rel_id = db.get_relation_id(request.relation)

        if head_id is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Entity '{request.head_entity}' not found"
            )
        if rel_id is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Relation '{request.relation}' not found"
            )

        # Get graph data
        edge_index, edge_type = db.get_graph_data()

        # Make prediction
        with torch.no_grad():
            if request.tail_entity:
                # Specific prediction
                tail_id = db.get_entity_id(request.tail_entity)
                if tail_id is None:
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Entity '{request.tail_entity}' not found"
                    )

                score = model.predict_links(
                    torch.tensor([head_id]),
                    torch.tensor([rel_id]),
                    torch.tensor([tail_id]),
                    edge_index=edge_index,
                    edge_type=edge_type
                )

                predictions = [PredictionResult(
                    entity=request.tail_entity,
                    entity_id=tail_id,
                    score=float(score[0]),
                    rank=1
                )]
                confidence = torch.sigmoid(score[0]).item()

            else:
                # Ranking prediction
                scores = model.predict_links(
                    torch.tensor([head_id]),
                    torch.tensor([rel_id]),
                    edge_index=edge_index,
                    edge_type=edge_type
                )

                # Get top-k predictions
                top_scores, top_indices = torch.topk(scores[0], request.top_k)

                predictions = []
                for rank, (idx, score) in enumerate(zip(top_indices, top_scores)):
                    entity_name = db.get_entity_name(idx.item())
                    predictions.append(PredictionResult(
                        entity=entity_name,
                        entity_id=idx.item(),
                        score=float(score),
                        rank=rank + 1
                    ))

                confidence = torch.sigmoid(top_scores[0]).item()

        # Get reasoning paths if requested
        reasoning_paths = None
        if request.include_reasoning and hasattr(model, 'get_reasoning_paths'):
            try:
                reasoning_paths = model.get_reasoning_paths(
                    head_id, rel_id, edge_index, edge_type, top_k=request.top_k
                )
            except Exception as e:
                logger.warning(f"Failed to get reasoning paths: {e}")

        processing_time = (time.time() - start_time) * 1000

        return LinkPredictionResponse(
            head_entity=request.head_entity,
            relation=request.relation,
            predictions=predictions,
            reasoning_paths=reasoning_paths,
            model_confidence=confidence,
            processing_time_ms=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in link prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/similar/{entity_name}")
async def find_similar_entities(
    entity_name: str,
    top_k: int = 10,
    model=Depends(lambda: None),
    db=Depends(lambda: None)
):
    """Find entities similar to the given entity based on embeddings."""
    try:
        entity_id = db.get_entity_id(entity_name)
        if entity_id is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Entity '{entity_name}' not found"
            )

        # Get graph data and entity embeddings
        edge_index, edge_type = db.get_graph_data()

        with torch.no_grad():
            entity_embs = model.forward(edge_index, edge_type)
            target_emb = entity_embs[entity_id].unsqueeze(0)

            # Calculate cosine similarity
            similarities = torch.cosine_similarity(target_emb, entity_embs)

            # Exclude the target entity
            similarities[entity_id] = -1
            top_similarities, top_indices = torch.topk(similarities, top_k)

        similar_entities = []
        for idx, sim in zip(top_indices, top_similarities):
            entity_name_similar = db.get_entity_name(idx.item())
            similar_entities.append({
                "entity": entity_name_similar,
                "entity_id": idx.item(),
                "similarity": float(sim)
            })

        return {
            "target_entity": entity_name,
            "similar_entities": similar_entities
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/relations/{entity_name}")
async def get_entity_relations(
    entity_name: str,
    db=Depends(lambda: None)
):
    """Get all relations connected to a specific entity."""
    try:
        entity_id = db.get_entity_id(entity_name)
        if entity_id is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Entity '{entity_name}' not found"
            )

        relations = db.get_entity_relations(entity_id)

        return {
            "entity": entity_name,
            "relations": relations
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entity relations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
