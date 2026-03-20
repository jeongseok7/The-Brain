import logging
from neo4j import AsyncGraphDatabase

_logger = logging.getLogger(__name__)


class Neo4jManager:
    def __init__(self, uri: str, username: str, password: str, database: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None

    def connect(self):
        """Initializes the Neo4j async driver."""
        self.driver = AsyncGraphDatabase.driver(
            self.uri, auth=(self.username, self.password)
        )
        _logger.info("Neo4j driver initialized.")

    async def close(self):
        """Closes the driver connection safely."""
        if self.driver:
            await self.driver.close()
            _logger.info("Neo4j driver closed.")

    async def get_stats(self) -> tuple[int, int]:
        """Returns (total_nodes, total_relations)."""
        if not self.driver:
            return 0, 0

        total_nodes = total_relations = 0
        try:
            async with self.driver.session(database=self.database) as session:
                r1 = await session.run("MATCH (n) RETURN count(n) AS c")
                total_nodes = (await r1.single())["c"]
                r2 = await session.run("MATCH ()-[r]->() RETURN count(r) AS c")
                total_relations = (await r2.single())["c"]
        except Exception as e:
            _logger.warning(f"Neo4j stats query failed: {e}")

        return total_nodes, total_relations

    async def get_graph(self, limit: int = 300, search: str = "") -> dict:
        """Fetches nodes and links for the 3D graph visualization."""
        if not self.driver:
            return {"nodes": [], "links": []}

        try:
            async with self.driver.session(database=self.database) as session:
                if search.strip():
                    cypher = """
                        MATCH (n)
                        WHERE toLower(n.entity_id) CONTAINS toLower($search)
                           OR toLower(coalesce(n.description,'')) CONTAINS toLower($search)
                        WITH n LIMIT 5
                        MATCH path = (n)-[r*0..2]-(neighbor)
                        WITH collect(DISTINCT startNode(relationships(path)[0])) +
                             collect(DISTINCT endNode(relationships(path)[0])) +
                             collect(DISTINCT n) AS allNodes,
                             collect(DISTINCT r) AS allRels
                        UNWIND allNodes AS node
                        WITH collect(DISTINCT node)[..200] AS topNodes
                        MATCH (a)-[r]->(b)
                        WHERE a IN topNodes AND b IN topNodes
                        RETURN
                          a.entity_id AS src, a.entity_type AS src_type,
                          coalesce(a.description,'') AS src_desc,
                          b.entity_id AS tgt, b.entity_type AS tgt_type,
                          coalesce(b.description,'') AS tgt_desc,
                          coalesce(r.description, type(r)) AS rel_label,
                          coalesce(r.weight, 1.0) AS weight
                        LIMIT 2000
                    """
                    result = await session.run(cypher, search=search.strip())
                else:
                    cypher = """
                        MATCH (n)
                        WITH n, size([(n)--() | 1]) AS degree
                        ORDER BY degree DESC
                        LIMIT $limit
                        WITH collect(n) AS topNodes
                        MATCH (a)-[r]->(b)
                        WHERE a IN topNodes AND b IN topNodes
                        RETURN
                          a.entity_id AS src, a.entity_type AS src_type,
                          coalesce(a.description,'') AS src_desc,
                          b.entity_id AS tgt, b.entity_type AS tgt_type,
                          coalesce(b.description,'') AS tgt_desc,
                          coalesce(r.description, type(r)) AS rel_label,
                          coalesce(r.weight, 1.0) AS weight
                        LIMIT 3000
                    """
                    result = await session.run(cypher, limit=limit)

                nodes: dict[str, dict] = {}
                links: list[dict] = []

                async for row in result:
                    src, tgt = row["src"], row["tgt"]
                    if not src or not tgt:
                        continue

                    if src not in nodes:
                        nodes[src] = {
                            "id": src,
                            "type": (row["src_type"] or "unknown").lower(),
                            "desc": row["src_desc"][:200] if row["src_desc"] else "",
                            "degree": 0,
                        }
                    if tgt not in nodes:
                        nodes[tgt] = {
                            "id": tgt,
                            "type": (row["tgt_type"] or "unknown").lower(),
                            "desc": row["tgt_desc"][:200] if row["tgt_desc"] else "",
                            "degree": 0,
                        }

                    nodes[src]["degree"] += 1
                    nodes[tgt]["degree"] += 1
                    links.append(
                        {
                            "source": src,
                            "target": tgt,
                            "label": (row["rel_label"] or "")[:80],
                            "weight": float(row["weight"] or 1.0),
                        }
                    )

                return {"nodes": list(nodes.values()), "links": links}

        except Exception as e:
            _logger.error(f"Graph query failed: {e}")
            raise e

    async def resolve_node_ids(self, entity_names: list[str]) -> list[str]:
        """Look up entity_ids in Neo4j that match the captured entity names."""
        if not entity_names or not self.driver:
            return []

        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    """
                    UNWIND $names AS name
                    MATCH (n)
                    WHERE toLower(n.entity_id) = toLower(name)
                       OR toLower(n.entity_id) CONTAINS toLower(name)
                    RETURN DISTINCT n.entity_id AS eid
                    LIMIT 60
                    """,
                    names=entity_names,
                )
                ids = []
                async for row in result:
                    if row["eid"]:
                        ids.append(row["eid"])
                return ids
        except Exception as e:
            _logger.warning(f"Node ID resolution failed: {e}")
            return []
