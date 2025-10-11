"""
Logging configuration and utilities.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional
import yaml


def setup_logging(
    config_path: Optional[Path] = None,
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None
):
    """Setup logging configuration."""

    # Default logging configuration
    default_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            }
        },
        'loggers': {
            'kg_reasoner': {
                'level': 'DEBUG',
                'handlers': ['console'],
                'propagate': False
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console']
        }
    }

    # Load config from file if provided
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Failed to load logging config: {e}")
            config = default_config
    else:
        config = default_config

    # Add file handler if log_file is specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': str(log_file),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }

        # Add file handler to loggers
        for logger_config in config['loggers'].values():
            if 'handlers' in logger_config:
                logger_config['handlers'].append('file')

    # Override log level if specified
    if log_level:
        level = getattr(logging, log_level.upper(), logging.INFO)
        config['handlers']['console']['level'] = level
        if 'file' in config['handlers']:
            config['handlers']['file']['level'] = level

    # Apply configuration
    logging.config.dictConfig(config)

    # Log the setup
    logger = logging.getLogger('kg_reasoner')
    logger.info("Logging setup completed")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(f'kg_reasoner.{name}')
