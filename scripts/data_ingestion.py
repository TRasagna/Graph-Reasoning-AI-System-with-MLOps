#!/usr/bin/env python3
"""
Data ingestion script for KG Reasoning system.
"""

import argparse
import sys
from pathlib import Path
import os
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


from etl.neo4j_ingestion import setup_sample_database
from utils.logging import setup_logging, get_logger
from utils.config import Config

setup_logging()
logger = get_logger('scripts.ingest')


def main():
    parser = argparse.ArgumentParser(description="Ingest data into KG Reasoning system")

    parser.add_argument(
        "--dataset", 
        type=str, 
        default="sample",
        choices=["sample", "fb15k-237", "wn18rr"],
        help="Dataset to ingest"
    )

    parser.add_argument(
        "--neo4j-uri", 
        type=str, 
        default=None,
        help="Neo4j URI (default from config/env)"
    )

    parser.add_argument(
        "--neo4j-username", 
        type=str, 
        default=None,
        help="Neo4j username (default from config/env)"
    )

    parser.add_argument(
        "--neo4j-password", 
        type=str, 
        default=None,
        help="Neo4j password (default from config/env)"
    )

    parser.add_argument(
        "--clear-existing", 
        action="store_true",
        help="Clear existing data before ingesting"
    )

    args = parser.parse_args()

    # Get database connection details
    config = Config()

    neo4j_uri = args.neo4j_uri or config.database.neo4j_uri
    neo4j_username = args.neo4j_username or config.database.neo4j_username
    neo4j_password = args.neo4j_password or config.database.neo4j_password

    logger.info(f"Ingesting dataset: {args.dataset}")
    logger.info(f"Neo4j URI: {neo4j_uri}")

    try:
        if args.dataset == "sample":
            stats = setup_sample_database(neo4j_uri, neo4j_username, neo4j_password)
            logger.info(f"Sample database created: {stats}")

        elif args.dataset == "fb15k-237":
            from etl.fb15k_ingestion import ingest_fb15k_237
            dataset_dir = Path("data/raw/fb15k-237") 
            ingest_fb15k_237(
                neo4j_uri,
                neo4j_username,
                neo4j_password,
                dataset_dir,
                clear_existing=args.clear_existing,
            )


        elif args.dataset == "wn18rr":
            logger.info("WN18RR ingestion not implemented yet") 
            return 1

        logger.info("Data ingestion completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
