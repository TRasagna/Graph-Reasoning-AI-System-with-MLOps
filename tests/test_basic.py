"""
Basic test to verify the system works.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from models.rgcn import RGCNModel
from etl.neo4j_ingestion import create_sample_knowledge_graph


def test_config_loading():
    """Test configuration loading."""
    config = Config()
    assert config.model.architecture in ['r-gcn', 'graph-transformer']
    assert config.model.embedding_dim > 0


def test_model_creation():
    """Test model creation."""
    model = RGCNModel(
        num_entities=100,
        num_relations=10,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=2
    )

    assert model.num_entities == 100
    assert model.num_relations == 10
    assert model.count_parameters() > 0


def test_sample_data_generation():
    """Test sample data generation."""
    triples, entity_info = create_sample_knowledge_graph()

    assert len(triples) > 0
    assert len(entity_info) > 0
    assert all(len(triple) == 3 for triple in triples)


if __name__ == "__main__":
    pytest.main([__file__])
