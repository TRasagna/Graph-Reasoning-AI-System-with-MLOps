import os
from neo4j import GraphDatabase
from utils.logging import get_logger

logger = get_logger("etl.fb15k")

def ingest_fb15k_237(uri, user, password, dataset_path, clear_existing=False):
    """
    Ingest FB15k-237 dataset into Neo4j.
    Expected folder structure:
        dataset_path/
            train.txt
            valid.txt
            test.txt
    Each line: head<TAB>relation<TAB>tail
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    def clear_db(session):
        logger.info("Clearing existing Neo4j data...")
        session.run("MATCH (n) DETACH DELETE n")

    def ingest_file(session, filepath, split_name):
        logger.info(f"Ingesting {split_name} data from {os.path.basename(filepath)}...")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                head, relation, tail = parts
                session.run(
                    """
                    MERGE (h:Entity {name: $head})
                    MERGE (t:Entity {name: $tail})
                    MERGE (h)-[r:RELATION {type: $rel, split: $split}]->(t)
                    """,
                    head=head, tail=tail, rel=relation, split=split_name
                )

    with driver.session() as session:
        if clear_existing:
            clear_db(session)

        for split_name in ["train", "valid", "test"]:
            file_path = os.path.join(dataset_path, f"{split_name}.txt")
            if os.path.exists(file_path):
                ingest_file(session, file_path, split_name)
            else:
                logger.warning(f"File not found: {file_path}")

    logger.info("FB15k-237 ingestion completed successfully!")
    driver.close()
