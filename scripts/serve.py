#!/usr/bin/env python3
"""
API server script for KG Reasoning system.
"""

import argparse
import sys
from pathlib import Path
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logging import setup_logging, get_logger
from utils.config import Config

setup_logging()
logger = get_logger('scripts.serve')


def main():
    parser = argparse.ArgumentParser(description="Start KG Reasoning API server")

    parser.add_argument(
        "--host", 
        type=str, 
        default=None,
        help="Host address (default from config)"
    )

    parser.add_argument(
        "--port", 
        type=int, 
        default=None,
        help="Port number (default from config)"
    )

    parser.add_argument(
        "--workers", 
        type=int, 
        default=None,
        help="Number of workers (default from config)"
    )

    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Enable auto-reload for development"
    )

    parser.add_argument(
        "--log-level", 
        type=str, 
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level"
    )

    args = parser.parse_args()

    # Get configuration
    config = Config()

    host = args.host or config.api.host
    port = args.port or config.api.port
    workers = args.workers or config.api.workers

    logger.info(f"Starting KG Reasoning API server")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Workers: {workers}")

    try:
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            workers=1 if args.reload else workers,
            reload=args.reload,
            log_level=args.log_level,
            app_dir=str(Path(__file__).parent.parent / "src")
        )

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
