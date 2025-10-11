#!/usr/bin/env python3
"""
Training script for KG Reasoning models.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train.trainer import train_model
from utils.logging import setup_logging, get_logger
from utils.config import Config

setup_logging()
logger = get_logger('scripts.train')


def main():
    parser = argparse.ArgumentParser(description="Train KG Reasoning models")

    parser.add_argument(
        "--architecture", 
        type=str, 
        default="r-gcn",
        choices=["r-gcn", "graph-transformer"],
        help="Model architecture to train"
    )

    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="./data/processed",
        help="Directory containing training data"
    )

    parser.add_argument(
        "--model-dir", 
        type=str, 
        default="./models",
        help="Directory to save trained models"
    )

    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=512,
        help="Training batch size"
    )

    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=0.001,
        help="Learning rate"
    )

    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--use-pretraining", 
        action="store_true",
        help="Use self-supervised pretraining"
    )

    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training"
    )

    args = parser.parse_args()

    logger.info(f"Starting training with architecture: {args.architecture}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Epochs: {args.epochs}")

    try:
        results = train_model(
            architecture=args.architecture,
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            num_epochs=args.epochs,
            config_path=args.config
        )

        logger.info(f"Training completed successfully!")
        logger.info(f"Model saved to: {results['model_path']}")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
