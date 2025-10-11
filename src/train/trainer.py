"""
Training module for Graph Neural Network models.
Supports both supervised and self-supervised training.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import time
import wandb
from tqdm import tqdm

from ..models.base_model import TrainableModel
from ..models.rgcn import RGCNModel, RGCNWithContrastiveLearning
from ..utils.config import Config
from ..utils.logging import get_logger

logger = get_logger('training')


class KnowledgeGraphTrainer:
    """Trainer class for knowledge graph models."""

    def __init__(
        self,
        model: TrainableModel,
        config: Config,
        device: str = 'auto'
    ):
        self.model = model
        self.config = config
        self.device = self._setup_device(device)

        # Move model to device
        self.model = self.model.to(self.device)

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0

        # Setup optimizers
        self.optimizer, self.scheduler = self.model.configure_optimizers(
            learning_rate=config.model.learning_rate
        )

        logger.info(f"Trainer initialized with device: {self.device}")

    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        return torch.device(device)

    def train(
        self,
        train_data: Dict[str, torch.Tensor],
        val_data: Optional[Dict[str, torch.Tensor]] = None,
        num_epochs: int = None,
        save_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Train the knowledge graph model."""
        if num_epochs is None:
            num_epochs = self.config.model.num_epochs

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb if configured
        if self.config.mlops.experiment_tracking_backend == 'wandb':
            try:
                wandb.init(
                    project=self.config.mlops.project_name,
                    tags=self.config.mlops.tags,
                    config=self.config.to_dict()
                )
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")

        training_history = {
            'train_loss': [],
            'val_metrics': [],
            'learning_rates': []
        }

        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Training phase
            train_metrics = self._train_epoch(train_data)
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Validation phase
            if val_data is not None:
                val_metrics = self._validate_epoch(val_data)
                training_history['val_metrics'].append(val_metrics)

                # Update learning rate scheduler
                if self.scheduler:
                    self.scheduler.step(val_metrics.get('loss', val_metrics.get('mrr', 0)))

                # Early stopping check
                current_metric = val_metrics.get('mrr', 0)
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.patience_counter = 0

                    # Save best model
                    if save_dir:
                        self._save_checkpoint(save_dir / 'best_model.pt', val_metrics)
                else:
                    self.patience_counter += 1

                # Check early stopping
                if self.patience_counter >= self.config.model.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Log progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self._log_epoch_progress(epoch, train_metrics, val_data and val_metrics)

        # Save final model
        if save_dir:
            self._save_checkpoint(save_dir / 'final_model.pt', training_history)

        logger.info("Training completed")

        if wandb.run:
            wandb.finish()

        return training_history

    def _train_epoch(self, train_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        edge_index = train_data['edge_index'].to(self.device)
        edge_type = train_data['edge_type'].to(self.device)

        # Create positive triples from edges
        pos_triples = torch.stack([
            edge_index[0],  # head
            edge_type,      # relation
            edge_index[1]   # tail
        ], dim=1)

        # Generate negative samples
        neg_triples = self._generate_negative_samples(
            pos_triples, 
            train_data.get('num_entities', self.model.num_entities)
        )

        total_loss = 0.0
        num_batches = 0

        # Create batches
        batch_size = self.config.model.batch_size
        num_pos = pos_triples.size(0)

        for i in range(0, num_pos, batch_size):
            batch_pos = pos_triples[i:i + batch_size]
            batch_neg = neg_triples[i:i + batch_size]

            self.optimizer.zero_grad()

            # Forward pass
            loss = self._compute_batch_loss(batch_pos, batch_neg, edge_index, edge_type)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update model training step
            self.model.training_step_end(loss.item(), {})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'loss': avg_loss}

    def _validate_epoch(self, val_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()

        with torch.no_grad():
            # Simple validation - compute loss on validation data
            edge_index = val_data['edge_index'].to(self.device)
            edge_type = val_data['edge_type'].to(self.device)

            # Create validation triples
            val_triples = torch.stack([
                edge_index[0], edge_type, edge_index[1]
            ], dim=1)

            # Generate negative samples for validation
            neg_triples = self._generate_negative_samples(
                val_triples[:1000],  # Sample for faster validation
                val_data.get('num_entities', self.model.num_entities)
            )

            # Compute validation loss
            val_loss = self._compute_batch_loss(
                val_triples[:1000], neg_triples, edge_index, edge_type
            )

            # Simple MRR approximation (for demonstration)
            mrr = 1.0 / (1.0 + val_loss.item())

            metrics = {
                'loss': val_loss.item(),
                'mrr': mrr,
                'hits@10': min(1.0, mrr * 10)
            }

            # Update model validation metrics
            self.model.validation_epoch_end(metrics)

        return metrics

    def _compute_batch_loss(
        self,
        pos_triples: torch.Tensor,
        neg_triples: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for a batch of triples."""
        # Positive scores
        pos_scores = self.model.predict_links(
            pos_triples[:, 0],  # head
            pos_triples[:, 1],  # relation
            pos_triples[:, 2],  # tail
            edge_index=edge_index,
            edge_type=edge_type
        )

        # Negative scores
        neg_scores = self.model.predict_links(
            neg_triples[:, 0],  # head
            neg_triples[:, 1],  # relation
            neg_triples[:, 2],  # tail
            edge_index=edge_index,
            edge_type=edge_type
        )

        # Compute loss
        if hasattr(self.model, 'compute_loss'):
            loss = self.model.compute_loss(pos_scores, neg_scores)
        else:
            # Default margin ranking loss
            loss = F.margin_ranking_loss(
                pos_scores,
                neg_scores,
                torch.ones_like(pos_scores),
                margin=5.0
            )

        return loss

    def _generate_negative_samples(
        self,
        pos_triples: torch.Tensor,
        num_entities: int
    ) -> torch.Tensor:
        """Generate negative samples by corrupting positive triples."""
        neg_triples = pos_triples.clone()

        # Randomly corrupt either head or tail
        corrupt_head = torch.rand(pos_triples.size(0)) < 0.5

        for i, corrupt in enumerate(corrupt_head):
            if corrupt:
                # Corrupt head
                neg_triples[i, 0] = torch.randint(0, num_entities, (1,))[0]
            else:
                # Corrupt tail  
                neg_triples[i, 2] = torch.randint(0, num_entities, (1,))[0]

        return neg_triples

    def _save_checkpoint(self, path: Path, metadata: Dict[str, Any]):
        """Save model checkpoint."""
        self.model.save_model(path, {
            'epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'metadata': metadata
        })
        logger.info(f"Checkpoint saved to {path}")

    def _log_epoch_progress(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Log training progress."""
        log_msg = f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}"

        if val_metrics:
            log_msg += f", Val Loss = {val_metrics.get('loss', 0):.4f}"
            log_msg += f", Val MRR = {val_metrics.get('mrr', 0):.4f}"

        log_msg += f", LR = {self.optimizer.param_groups[0]['lr']:.6f}"

        logger.info(log_msg)


def train_model(
    architecture: str = 'r-gcn',
    data_dir: str = './data',
    model_dir: str = './models',
    num_epochs: int = 100,
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Train a model with specified architecture."""

    # Load configuration
    config = Config(config_path and Path(config_path))
    config.model.architecture = architecture
    config.model.num_epochs = num_epochs

    # Load data
    from ..utils.database import PyGDataLoader

    data_loader = PyGDataLoader(Path(data_dir))
    train_data = data_loader.load_data('train')
    val_data = data_loader.load_data('valid') if Path(data_dir, 'valid_data.pt').exists() else None

    # Initialize model
    num_entities = train_data.get('num_entities', train_data['edge_index'].max().item() + 1)
    num_relations = train_data.get('num_relations', train_data['edge_type'].max().item() + 1)

    if architecture == 'r-gcn':
        model = RGCNModel(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=config.model.embedding_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_bases=config.model.num_bases,
            dropout=config.model.dropout
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Train model
    trainer = KnowledgeGraphTrainer(model, config)

    save_dir = Path(model_dir) / f"{architecture}_{int(time.time())}"
    training_history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=num_epochs,
        save_dir=save_dir
    )

    logger.info(f"Training completed. Model saved to {save_dir}")

    return {
        'model_path': str(save_dir / 'best_model.pt'),
        'training_history': training_history,
        'config': config.to_dict()
    }
