"""
Base model class for all graph neural network models.
Provides common functionality and interfaces.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import wandb

logger = logging.getLogger(__name__)


class BaseGraphModel(nn.Module, ABC):
    """Abstract base class for all graph neural network models."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        **kwargs
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # Initialize entity and relation embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings with Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    @abstractmethod
    def forward(
        self, 
        edge_index: torch.Tensor, 
        edge_type: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass of the model."""
        pass

    @abstractmethod
    def predict_links(
        self,
        head_entities: torch.Tensor,
        relations: torch.Tensor,
        tail_entities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict link probabilities."""
        pass

    def save_model(self, path: Path, metadata: Optional[Dict[str, Any]] = None):
        """Save model state with metadata."""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'num_entities': self.num_entities,
                'num_relations': self.num_relations,
                'embedding_dim': self.embedding_dim
            }
        }
        if metadata:
            save_dict['metadata'] = metadata

        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: Path, **kwargs):
        """Load model from saved state."""
        checkpoint = torch.load(path, map_location='cpu')
        model_config = checkpoint['model_config']

        # Override with any provided kwargs
        model_config.update(kwargs)

        model = cls(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Model loaded from {path}")
        return model, checkpoint.get('metadata', {})

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current entity and relation embeddings."""
        return self.entity_embeddings.weight.data, self.relation_embeddings.weight.data


class TrainableModel(BaseGraphModel):
    """Extended base class with training utilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_step = 0
        self.validation_metrics = {}

    def training_step_end(self, loss: float, metrics: Dict[str, float]):
        """Called at the end of each training step."""
        self.training_step += 1

        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                'train/loss': loss,
                'train/step': self.training_step,
                **{f'train/{k}': v for k, v in metrics.items()}
            })

    def validation_epoch_end(self, metrics: Dict[str, float]):
        """Called at the end of each validation epoch."""
        self.validation_metrics = metrics

        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                f'val/{k}': v for k, v in metrics.items()
            })

    def configure_optimizers(self, learning_rate: float = 1e-3, weight_decay: float = 1e-4):
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

        return optimizer, scheduler
