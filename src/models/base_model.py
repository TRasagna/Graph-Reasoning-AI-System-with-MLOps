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

        # Defensive: if the saved state_dict encodes a different number of
        # relation types (e.g., due to mismatched DB mappings at save/load),
        # prefer the shape encoded in the state_dict so we can instantiate a
        # model with matching parameter shapes before calling load_state_dict.
        try:
            state = checkpoint.get('model_state_dict', {})
            # Try relation_embeddings first (most explicit)
            rel_key = None
            for k in state.keys():
                if k.endswith('relation_embeddings.weight') or k == 'relation_embeddings.weight':
                    rel_key = k
                    break

            inferred_relations = None
            if rel_key is not None:
                inferred_relations = state[rel_key].size(0)
            else:
                # Fallback: inspect RGCN internal 'comp' parameter shapes which
                # are typically [num_relations, num_bases]
                for k in state.keys():
                    if '.comp' in k and 'rgcn_layers' in k:
                        try:
                            inferred_relations = state[k].size(0)
                            break
                        except Exception:
                            continue

            if inferred_relations is not None and model_config.get('num_relations') != inferred_relations:
                logger.warning(
                    f"Checkpoint model_config.num_relations={model_config.get('num_relations')} "
                    f"differs from state_dict-inferred relations={inferred_relations}. "
                    "Overriding model_config to match the checkpoint shapes to allow loading."
                )
                model_config['num_relations'] = int(inferred_relations)
        except Exception as e:
            logger.warning(f"Failed to infer relation count from checkpoint state: {e}")

        model = cls(**model_config)
        state_dict = checkpoint['model_state_dict']

        # First try strict loading. If that fails due to shape mismatches
        # (common when DB mappings differ from training), attempt a non-strict
        # load so compatible parameters are restored and incompatible ones
        # remain initialized.
        try:
            model.load_state_dict(state_dict)
            logger.info(f"Model loaded from {path} (strict)")
        except RuntimeError as e:
            logger.warning(f"Strict state_dict load failed: {e}")
            try:
                load_result = model.load_state_dict(state_dict, strict=False)
                # load_result is a NamedTuple with missing_keys and unexpected_keys
                missing = getattr(load_result, 'missing_keys', None)
                unexpected = getattr(load_result, 'unexpected_keys', None)
                logger.info(f"Model partially loaded from {path} (non-strict). Missing keys: {missing}; Unexpected keys: {unexpected}")
            except Exception as e2:
                logger.error(f"Failed to load state_dict even with strict=False: {e2}")
                raise

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
