"""
Database connection and utility functions.
"""

import torch
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class Neo4jConnection:
    """Neo4j database connection and query utilities."""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._driver = None
        self._entity_to_id = {}
        self._id_to_entity = {}
        self._relation_to_id = {}
        self._id_to_relation = {}
        self.connect()

    def connect(self):
        """Establish connection to Neo4j."""
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            with self._driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j successfully")
            self._build_mappings()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close database connection."""
        if self._driver:
            self._driver.close()

    def _build_mappings(self):
        """Build entity and relation ID mappings."""
        try:
            with self._driver.session(database=self.database) as session:
                # Build entity mappings
                result = session.run("MATCH (e:Entity) RETURN e.id as entity_id ORDER BY e.id")
                entities = [record["entity_id"] for record in result]

                self._entity_to_id = {entity: idx for idx, entity in enumerate(entities)}
                self._id_to_entity = {idx: entity for idx, entity in enumerate(entities)}

                # Build relation mappings
                result = session.run("""
                    MATCH ()-[r:RELATION]->()
                    RETURN DISTINCT r.type as relation_type
                    ORDER BY r.type
                """)
                relations = [record["relation_type"] for record in result]

                self._relation_to_id = {relation: idx for idx, relation in enumerate(relations)}
                self._id_to_relation = {idx: relation for idx, relation in enumerate(relations)}

                logger.info(f"Built mappings: {len(entities)} entities, {len(relations)} relations")

        except Exception as e:
            logger.warning(f"Failed to build ID mappings: {e}")

    def get_entity_id(self, entity_name: str) -> Optional[int]:
        """Get entity ID from name."""
        return self._entity_to_id.get(entity_name)

    def get_entity_name(self, entity_id: int) -> Optional[str]:
        """Get entity name from ID."""
        return self._id_to_entity.get(entity_id)

    def get_relation_id(self, relation_name: str) -> Optional[int]:
        """Get relation ID from name."""
        return self._relation_to_id.get(relation_name)

    def get_relation_name(self, relation_id: int) -> Optional[str]:
        """Get relation name from ID."""
        return self._id_to_relation.get(relation_id)

    def get_graph_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get graph data in PyTorch Geometric format."""
        try:
            with self._driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (h:Entity)-[r:RELATION]->(t:Entity)
                    RETURN h.id as head, r.type as relation, t.id as tail
                """)

                edges = []
                edge_types = []

                for record in result:
                    head_id = self.get_entity_id(record["head"])
                    tail_id = self.get_entity_id(record["tail"])
                    relation_id = self.get_relation_id(record["relation"])

                    if head_id is not None and tail_id is not None and relation_id is not None:
                        edges.append([head_id, tail_id])
                        edge_types.append(relation_id)

                if edges:
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                    edge_type = torch.tensor(edge_types, dtype=torch.long)
                    return edge_index, edge_type
                else:
                    # Return empty tensors if no data
                    return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.long)

        except Exception as e:
            logger.error(f"Failed to get graph data: {e}")
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.long)

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get basic graph statistics."""
        with self._driver.session(database=self.database) as session:
            stats = {}

            # Count entities
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            stats['num_entities'] = result.single()["count"]

            # Count relationships
            result = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) as count")
            stats['num_relationships'] = result.single()["count"]

            # Count relation types
            result = session.run("MATCH ()-[r:RELATION]->() RETURN count(DISTINCT r.type) as count")
            stats['num_relation_types'] = result.single()["count"]

            return stats

    def get_entity_relations(self, entity_id: int) -> List[Dict[str, Any]]:
        """Get all relations for a specific entity."""
        entity_name = self.get_entity_name(entity_id)
        if not entity_name:
            return []

        with self._driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (e:Entity {id: $entity_name})-[r:RELATION]-(other:Entity)
                RETURN r.type as relation, other.id as connected_entity,
                       CASE WHEN startNode(r) = e THEN 'outgoing' ELSE 'incoming' END as direction
                LIMIT 100
            """, entity_name=entity_name)

            relations = []
            for record in result:
                relations.append({
                    'relation': record['relation'],
                    'connected_entity': record['connected_entity'],
                    'direction': record['direction']
                })

            return relations


class PyGDataLoader:
    """Data loader for PyTorch Geometric format."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_cache = {}

    def load_data(self, split: str = 'train') -> Dict[str, torch.Tensor]:
        """Load data for a specific split."""
        if split in self.data_cache:
            return self.data_cache[split]

        data_file = self.data_dir / f"{split}_data.pt"

        if data_file.exists():
            data = torch.load(data_file)
            self.data_cache[split] = data
            return data
        else:
            # Create dummy data for testing
            logger.warning(f"Data file {data_file} not found, creating dummy data")
            return self._create_dummy_data()

    def _create_dummy_data(self) -> Dict[str, torch.Tensor]:
        """Create dummy data for testing purposes."""
        num_entities = 1000
        num_relations = 50
        num_edges = 5000

        # Create random edges
        edge_index = torch.randint(0, num_entities, (2, num_edges))
        edge_type = torch.randint(0, num_relations, (num_edges,))

        # Create some triples for evaluation
        num_triples = 1000
        triples = torch.randint(0, min(num_entities, num_relations), (num_triples, 3))

        return {
            'edge_index': edge_index,
            'edge_type': edge_type,
            'triples': triples,
            'num_entities': num_entities,
            'num_relations': num_relations
        }

    def save_data(self, data: Dict[str, torch.Tensor], split: str = 'train'):
        """Save data to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data_file = self.data_dir / f"{split}_data.pt"
        torch.save(data, data_file)
        logger.info(f"Data saved to {data_file}")
