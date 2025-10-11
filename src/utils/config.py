"""
Configuration management utilities.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password123"
    neo4j_database: str = "neo4j"


@dataclass
class ModelConfig:
    architecture: str = "r-gcn"
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    num_bases: int = 100
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 512
    num_epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class TrainingConfig:
    pretrain_epochs: int = 50
    finetune_epochs: int = 100
    validation_split: float = 0.1
    test_split: float = 0.1
    contrastive_temperature: float = 0.1
    mask_ratio: float = 0.15


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: list = None
    max_concurrent_requests: int = 100

    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


@dataclass
class MLOpsConfig:
    experiment_tracking_backend: str = "wandb"
    project_name: str = "kg-reasoning-system"
    tags: list = None
    model_registry_backend: str = "wandb"
    model_name: str = "kg-reasoner"

    def __post_init__(self):
        if self.tags is None:
            self.tags = ["graph-neural-networks", "knowledge-graphs", "reasoning"]


@dataclass
class PathsConfig:
    data_dir: str = "./data"
    models_dir: str = "./models"
    logs_dir: str = "./logs"
    artifacts_dir: str = "./artifacts"


class Config:
    """Main configuration class that loads and manages all settings."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/config.yaml")

        # Initialize configs with defaults
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.api = APIConfig()
        self.mlops = MLOpsConfig()
        self.paths = PathsConfig()

        # Load from file if exists
        if self.config_path.exists():
            self.load_from_file()

        # Override with environment variables
        self.load_from_env()

        logger.info(f"Configuration loaded from {self.config_path}")

    def load_from_file(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            if 'database' in config_data:
                self._update_dataclass(self.database, config_data['database'])
            if 'model' in config_data:
                self._update_dataclass(self.model, config_data['model'])
            if 'training' in config_data:
                self._update_dataclass(self.training, config_data['training'])
            if 'api' in config_data:
                self._update_dataclass(self.api, config_data['api'])
            if 'mlops' in config_data:
                self._update_dataclass(self.mlops, config_data['mlops'])
            if 'paths' in config_data:
                self._update_dataclass(self.paths, config_data['paths'])

        except Exception as e:
            logger.warning(f"Failed to load config from file: {e}")

    def load_from_env(self):
        """Load configuration from environment variables."""
        # Database
        self.database.neo4j_uri = os.getenv('NEO4J_URI', self.database.neo4j_uri)
        self.database.neo4j_username = os.getenv('NEO4J_USERNAME', self.database.neo4j_username)
        self.database.neo4j_password = os.getenv('NEO4J_PASSWORD', self.database.neo4j_password)
        self.database.neo4j_database = os.getenv('NEO4J_DATABASE', self.database.neo4j_database)

        # Model
        self.model.architecture = os.getenv('MODEL_ARCHITECTURE', self.model.architecture)
        self.model.batch_size = int(os.getenv('BATCH_SIZE', self.model.batch_size))
        self.model.learning_rate = float(os.getenv('LEARNING_RATE', self.model.learning_rate))

        # API
        self.api.host = os.getenv('API_HOST', self.api.host)
        self.api.port = int(os.getenv('API_PORT', self.api.port))
        self.api.workers = int(os.getenv('API_WORKERS', self.api.workers))

        # MLOps
        self.mlops.project_name = os.getenv('WANDB_PROJECT', self.mlops.project_name)

    @staticmethod
    def _update_dataclass(dataclass_obj, update_dict):
        """Update dataclass with dictionary values."""
        for key, value in update_dict.items():
            if hasattr(dataclass_obj, key):
                setattr(dataclass_obj, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'database': asdict(self.database),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'api': asdict(self.api),
            'mlops': asdict(self.mlops),
            'paths': asdict(self.paths)
        }

    def save_to_file(self, path: Optional[Path] = None):
        """Save current configuration to file."""
        save_path = path or self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {save_path}")


def get_config() -> Config:
    """Get the global configuration instance."""
    return Config()
