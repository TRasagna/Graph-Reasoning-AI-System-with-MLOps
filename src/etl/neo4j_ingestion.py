"""
Neo4j ETL pipeline for knowledge graph data ingestion.
Handles FB15k-237, WN18RR, and custom datasets.
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
from tqdm import tqdm
import time

from ..utils.logging import get_logger

logger = get_logger('etl')


class Neo4jKnowledgeGraphBuilder:
    """Build and manage knowledge graphs in Neo4j."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j"
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        self.connect()

    def connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j database successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def clear_database(self):
        """Clear all nodes and relationships."""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")

    def create_indexes(self):
        """Create necessary indexes for performance."""
        indexes = [
            "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX relation_type_index IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.type)"
        ]

        with self.driver.session(database=self.database) as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                    logger.info(f"Created index: {index_query.split()[2]}")
                except Exception as e:
                    logger.warning(f"Index creation failed: {e}")

    def ingest_triples(
        self,
        triples: List[Tuple[str, str, str]],
        entity_info: Optional[Dict[str, Dict]] = None,
        relation_info: Optional[Dict[str, Dict]] = None,
        batch_size: int = 10000
    ):
        """Ingest knowledge graph triples into Neo4j."""
        logger.info(f"Starting ingestion of {len(triples)} triples")

        # Collect unique entities and relations
        entities = set()
        relations = set()

        for head, relation, tail in triples:
            entities.add(head)
            entities.add(tail)
            relations.add(relation)

        logger.info(f"Found {len(entities)} unique entities and {len(relations)} unique relations")

        # Create entities
        self._create_entities(list(entities), entity_info, batch_size)

        # Create relations metadata
        self._create_relation_types(list(relations), relation_info, batch_size)

        # Create relationships
        self._create_relationships(triples, batch_size)

        logger.info("Triple ingestion completed")

    def _create_entities(
        self,
        entities: List[str],
        entity_info: Optional[Dict[str, Dict]] = None,
        batch_size: int = 10000
    ):
        """Create entity nodes in batches."""
        query = """
        UNWIND $entities AS entity_data
        MERGE (e:Entity {id: entity_data.id})
        SET e.name = entity_data.name,
            e.description = entity_data.description,
            e.type = entity_data.type,
            e.created_at = datetime()
        """

        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(entities), batch_size), desc="Creating entities"):
                batch_entities = entities[i:i + batch_size]
                entity_batch = []

                for entity_id in batch_entities:
                    entity_data = {
                        'id': entity_id,
                        'name': entity_id,  # Default name
                        'description': '',
                        'type': 'unknown'
                    }

                    # Add additional info if available
                    if entity_info and entity_id in entity_info:
                        entity_data.update(entity_info[entity_id])

                    entity_batch.append(entity_data)

                session.run(query, entities=entity_batch)

        logger.info(f"Created {len(entities)} entity nodes")

    def _create_relation_types(
        self,
        relations: List[str],
        relation_info: Optional[Dict[str, Dict]] = None,
        batch_size: int = 1000
    ):
        """Create relation type metadata."""
        query = """
        UNWIND $relations AS rel_data
        MERGE (r:RelationType {type: rel_data.type})
        SET r.name = rel_data.name,
            r.description = rel_data.description,
            r.created_at = datetime()
        """

        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(relations), batch_size), desc="Creating relation types"):
                batch_relations = relations[i:i + batch_size]
                relation_batch = []

                for relation_type in batch_relations:
                    rel_data = {
                        'type': relation_type,
                        'name': relation_type,
                        'description': ''
                    }

                    if relation_info and relation_type in relation_info:
                        rel_data.update(relation_info[relation_type])

                    relation_batch.append(rel_data)

                session.run(query, relations=relation_batch)

        logger.info(f"Created {len(relations)} relation types")

    def _create_relationships(
        self,
        triples: List[Tuple[str, str, str]],
        batch_size: int = 10000
    ):
        """Create relationships between entities."""
        query = """
        UNWIND $triples AS triple
        MATCH (h:Entity {id: triple.head})
        MATCH (t:Entity {id: triple.tail})
        MERGE (h)-[r:RELATION {type: triple.relation}]->(t)
        SET r.weight = coalesce(r.weight, 0) + 1,
            r.created_at = coalesce(r.created_at, datetime()),
            r.updated_at = datetime()
        """

        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(triples), batch_size), desc="Creating relationships"):
                batch_triples = triples[i:i + batch_size]
                triple_batch = [
                    {
                        'head': head,
                        'relation': relation,
                        'tail': tail
                    }
                    for head, relation, tail in batch_triples
                ]

                session.run(query, triples=triple_batch)

        logger.info(f"Created {len(triples)} relationships")

    def get_graph_statistics(self) -> Dict[str, int]:
        """Get basic statistics about the knowledge graph."""
        with self.driver.session(database=self.database) as session:
            # Count entities
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            entity_count = result.single()["count"]

            # Count relationships
            result = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) as count")
            rel_count = result.single()["count"]

            # Count unique relation types
            result = session.run(
                "MATCH ()-[r:RELATION]->() RETURN count(DISTINCT r.type) as count"
            )
            rel_type_count = result.single()["count"]

            return {
                'num_entities': entity_count,
                'num_relationships': rel_count,
                'num_relation_types': rel_type_count
            }

    def query_subgraph(
        self,
        center_entity: str,
        max_hops: int = 2,
        max_nodes: int = 100
    ) -> Dict[str, List]:
        """Extract subgraph around a center entity."""
        with self.driver.session(database=self.database) as session:
            query = f"""
                MATCH path = (center:Entity {{id: $center_entity}})-[*1..{max_hops}]-(neighbor:Entity)
                WITH collect(DISTINCT center) + collect(DISTINCT neighbor) as nodes,
                     collect(DISTINCT relationships(path)) as rels
                RETURN nodes[..{max_nodes}] as nodes,
                       rels[..{max_nodes * 2}] as relationships
            """

            result = session.run(query, center_entity=center_entity).single()

            if result:
                nodes = [{'id': node['id'], 'name': node.get('name', node['id'])} 
                        for node in result['nodes']]
                relationships = []

                for rel_list in result['relationships']:
                    for rel in rel_list:
                        relationships.append({
                            'head': rel.start_node['id'],
                            'relation': rel['type'],
                            'tail': rel.end_node['id']
                        })

                return {
                    'nodes': nodes,
                    'relationships': relationships
                }

        return {'nodes': [], 'relationships': []}


def create_sample_knowledge_graph():
    """Create a sample knowledge graph for testing."""
    sample_triples = [
        ("Albert_Einstein", "born_in", "Germany"),
        ("Germany", "located_in", "Europe"),
        ("Albert_Einstein", "developed", "Theory_of_Relativity"),
        ("Theory_of_Relativity", "is_a", "Physics_Theory"),
        ("Albert_Einstein", "won", "Nobel_Prize"),
        ("Nobel_Prize", "awarded_by", "Nobel_Committee"),
        ("Marie_Curie", "born_in", "Poland"),
        ("Poland", "located_in", "Europe"),
        ("Marie_Curie", "won", "Nobel_Prize"),
        ("Marie_Curie", "discovered", "Radium"),
        ("Radium", "is_a", "Chemical_Element"),
        ("Isaac_Newton", "born_in", "England"),
        ("England", "located_in", "Europe"),
        ("Isaac_Newton", "discovered", "Law_of_Gravity"),
        ("Law_of_Gravity", "is_a", "Physics_Law")
    ]

    entity_info = {
        "Albert_Einstein": {
            "name": "Albert Einstein",
            "description": "German-born theoretical physicist",
            "type": "Person"
        },
        "Marie_Curie": {
            "name": "Marie Curie",
            "description": "Polish physicist and chemist",
            "type": "Person"
        },
        "Isaac_Newton": {
            "name": "Isaac Newton",
            "description": "English mathematician and physicist",
            "type": "Person"
        },
        "Germany": {
            "name": "Germany",
            "description": "Country in Central Europe",
            "type": "Country"
        },
        "Poland": {
            "name": "Poland",
            "description": "Country in Central Europe",
            "type": "Country"
        },
        "England": {
            "name": "England",
            "description": "Country in the United Kingdom",
            "type": "Country"
        },
        "Europe": {
            "name": "Europe",
            "description": "Continent",
            "type": "Continent"
        }
    }

    return sample_triples, entity_info


def setup_sample_database(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_username: str = "neo4j",
    neo4j_password: str = "password123"
):
    """Setup a sample database for testing."""
    logger.info("Setting up sample knowledge graph database")

    kg_builder = Neo4jKnowledgeGraphBuilder(neo4j_uri, neo4j_username, neo4j_password)

    try:
        # Clear existing data
        kg_builder.clear_database()

        # Create indexes
        kg_builder.create_indexes()

        # Load sample data
        sample_triples, entity_info = create_sample_knowledge_graph()
        kg_builder.ingest_triples(sample_triples, entity_info)

        # Get statistics
        stats = kg_builder.get_graph_statistics()
        logger.info(f"Sample database created: {stats}")

        return stats

    finally:
        kg_builder.close()
