"""
Relational Graph Convolutional Network (R-GCN) implementation
for knowledge graph reasoning and link prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from typing import Optional, Tuple, Dict, Any
import logging

from src.models.base_model import TrainableModel

logger = logging.getLogger(__name__)


class RGCNModel(TrainableModel):
    """R-GCN model for knowledge graph reasoning."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_bases: int = 100,
        dropout: float = 0.2,
        use_self_loop: bool = True,
        layer_norm: bool = True,
        **kwargs
    ):
        super().__init__(num_entities, num_relations, embedding_dim, **kwargs)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_bases = num_bases
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.layer_norm = layer_norm

        # Build R-GCN layers
        self.rgcn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if layer_norm else None

        # Input layer
        self.rgcn_layers.append(
            RGCNConv(
                embedding_dim,
                hidden_dim,
                num_relations,
                num_bases=num_bases,
                root_weight=use_self_loop
            )
        )
        if layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.rgcn_layers.append(
                RGCNConv(
                    hidden_dim,
                    hidden_dim,
                    num_relations,
                    num_bases=num_bases,
                    root_weight=use_self_loop
                )
            )
            if layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Output layer
        if num_layers > 1:
            self.rgcn_layers.append(
                RGCNConv(
                    hidden_dim,
                    embedding_dim,
                    num_relations,
                    num_bases=num_bases,
                    root_weight=use_self_loop
                )
            )
            if layer_norm:
                self.layer_norms.append(nn.LayerNorm(embedding_dim))

        # Scoring layers for link prediction
        self.score_head = nn.Linear(embedding_dim, embedding_dim)
        self.score_tail = nn.Linear(embedding_dim, embedding_dim)
        self.score_relation = nn.Linear(embedding_dim, embedding_dim)

        self.dropout_layer = nn.Dropout(dropout)

        logger.info(f"Initialized R-GCN model with {self.count_parameters()} parameters")

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through R-GCN layers.

        Args:
            edge_index: Edge connectivity [2, num_edges]
            edge_type: Edge types [num_edges]
            entity_ids: Specific entities to compute embeddings for

        Returns:
            Entity embeddings [num_entities, embedding_dim]
        """
        # Get initial embeddings
        if entity_ids is not None:
            x = self.entity_embeddings(entity_ids)
        else:
            x = self.entity_embeddings.weight

        # Forward through R-GCN layers
        for i, layer in enumerate(self.rgcn_layers):
            x = layer(x, edge_index, edge_type)

            # Apply layer normalization
            if self.layer_norm and i < len(self.layer_norms):
                x = self.layer_norms[i](x)

            # Apply activation (except for last layer)
            if i < len(self.rgcn_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)

        return x

    def predict_links(
        self,
        head_entities: torch.Tensor,
        relations: torch.Tensor,
        tail_entities: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict link probabilities using TransE/DistMult scoring.

        Args:
            head_entities: Head entity IDs [batch_size]
            relations: Relation IDs [batch_size]
            tail_entities: Tail entity IDs [batch_size] (optional)
            edge_index: Full graph edge connectivity
            edge_type: Full graph edge types

        Returns:
            Scores [batch_size] or [batch_size, num_entities]
        """
        # Get entity embeddings from R-GCN
        if edge_index is not None and edge_type is not None:
            entity_embs = self.forward(edge_index, edge_type)
        else:
            entity_embs = self.entity_embeddings.weight

        # Get head and relation embeddings
        head_embs = entity_embs[head_entities]  # [batch_size, embedding_dim]
        rel_embs = self.relation_embeddings(relations)  # [batch_size, embedding_dim]

        # Transform embeddings for scoring
        head_embs = self.score_head(head_embs)
        rel_embs = self.score_relation(rel_embs)

        if tail_entities is not None:
            # Specific tail entities - return scores for given triples
            tail_embs = entity_embs[tail_entities]
            tail_embs = self.score_tail(tail_embs)

            # TransE scoring: -||h + r - t||
            scores = -torch.norm(head_embs + rel_embs - tail_embs, p=2, dim=1)
        else:
            # All possible tail entities - return scores for all
            all_tail_embs = self.score_tail(entity_embs)  # [num_entities, embedding_dim]

            # Batch computation
            head_rel = head_embs + rel_embs  # [batch_size, embedding_dim]
            head_rel = head_rel.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            all_tail_embs = all_tail_embs.unsqueeze(0)  # [1, num_entities, embedding_dim]

            # Compute distances
            distances = torch.norm(head_rel - all_tail_embs, p=2, dim=2)  # [batch_size, num_entities]
            scores = -distances

        return scores

    def compute_loss(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        margin: float = 5.0
    ) -> torch.Tensor:
        """Compute margin ranking loss for link prediction."""
        loss = F.margin_ranking_loss(
            pos_scores,
            neg_scores,
            torch.ones_like(pos_scores),
            margin=margin
        )
        return loss


class RGCNWithContrastiveLearning(RGCNModel):
    """R-GCN model with contrastive learning for self-supervised pretraining."""

    def __init__(self, *args, temperature: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2)
        )

    def contrastive_loss(
        self,
        anchor_embs: torch.Tensor,
        positive_embs: torch.Tensor,
        negative_embs: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss for self-supervised learning."""
        # Project embeddings
        anchor_proj = self.projection_head(anchor_embs)
        positive_proj = self.projection_head(positive_embs)
        negative_proj = self.projection_head(negative_embs)

        # Normalize embeddings
        anchor_proj = F.normalize(anchor_proj, dim=-1)
        positive_proj = F.normalize(positive_proj, dim=-1)
        negative_proj = F.normalize(negative_proj, dim=-1)

        # Compute similarity scores
        pos_sim = torch.sum(anchor_proj * positive_proj, dim=-1) / self.temperature
        neg_sim = torch.sum(anchor_proj * negative_proj, dim=-1) / self.temperature

        # Contrastive loss
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
        return loss.mean()
