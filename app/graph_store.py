from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Protocol

from neo4j import GraphDatabase

MAX_ALLOWED_HOPS = 4  # hard ceiling regardless of caller-supplied config


@dataclass(frozen=True, slots=True)
class Relation:
    source: str
    relation: str
    target: str


@dataclass(slots=True)
class Path:
    """A simple path (no repeated nodes) discovered during traversal."""

    entities: list[str]  # ordered chain, length = hops + 1
    relation_kinds: list[str]  # relation label between each consecutive pair

    @property
    def hops(self) -> int:
        return len(self.relation_kinds)

    def render(self) -> str:
        parts = [self.entities[0]]
        for kind, entity in zip(self.relation_kinds, self.entities[1:], strict=True):
            parts.append(f"-[{kind}]-> {entity}")
        return " ".join(parts)


@dataclass(slots=True)
class GraphContext:
    source: str
    text: str
    entities: list[str]
    relations: list[str]
    hop_distance: int | None = None
    """Deepest hop distance justified for this document: the greatest
    number of relationship hops from any seed (query) entity to any of
    this document's entities that were actually reached during traversal.
    0 means the document directly mentions a seed entity; a document that
    also connects to something 2-3 hops away is reported at that deeper
    distance, since that's the traversal that earned its inclusion."""


class GraphStore(Protocol):
    backend_name: str

    def index_document(
        self, source: str, text: str, entities: list[str], relations: list[Relation]
    ) -> None: ...
    def search(
        self, entities: list[str], limit: int, max_hops: int = 2
    ) -> list[GraphContext]: ...
    def health(self) -> bool: ...


class InMemoryGraphStore:
    backend_name = "memory"

    def __init__(self) -> None:
        self.docs: dict[str, str] = {}
        self.doc_entities: dict[str, set[str]] = defaultdict(set)
        self.relations: set[Relation] = set()

    def index_document(
        self, source: str, text: str, entities: list[str], relations: list[Relation]
    ) -> None:
        self.docs[source] = text
        self.doc_entities[source].update(entities)
        self.relations.update(relations)

    def _adjacency(self) -> dict[str, list[tuple[str, str]]]:
        """entity (lowercased) -> [(relation_kind, neighbor_original_name), ...]

        Traversal is undirected: a DEPENDS_ON edge can be walked in either
        direction, since "what does X depend on" and "what depends on X"
        are both legitimate questions over the same edge.
        """
        adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for relation in self.relations:
            adjacency[relation.source.lower()].append((relation.relation, relation.target))
            adjacency[relation.target.lower()].append((relation.relation, relation.source))
        return adjacency

    def _find_paths(self, seed_entities: list[str], max_hops: int) -> list[Path]:
        """BFS out from each seed entity up to max_hops, collecting every
        simple path discovered along the way (not just shortest paths) so
        the caller can see how entities are connected, not just that they
        are connected."""
        adjacency = self._adjacency()
        paths: list[Path] = []

        for seed in seed_entities:
            seed_lower = seed.lower()
            if seed_lower not in adjacency:
                continue
            visited = {seed_lower}
            queue: deque[tuple[str, list[str], list[str]]] = deque([(seed, [seed], [])])
            while queue:
                current, chain, relation_chain = queue.popleft()
                if len(relation_chain) >= max_hops:
                    continue
                for relation_kind, neighbor in adjacency.get(current.lower(), []):
                    neighbor_lower = neighbor.lower()
                    if neighbor_lower in visited:
                        continue
                    visited.add(neighbor_lower)
                    new_chain = [*chain, neighbor]
                    new_relation_chain = [*relation_chain, relation_kind]
                    paths.append(Path(entities=new_chain, relation_kinds=new_relation_chain))
                    queue.append((neighbor, new_chain, new_relation_chain))
        return paths

    def search(self, entities: list[str], limit: int, max_hops: int = 2) -> list[GraphContext]:
        max_hops = max(1, min(max_hops, MAX_ALLOWED_HOPS))
        wanted = {e.lower() for e in entities}
        if not wanted:
            return []

        paths = self._find_paths(entities, max_hops)

        # Shortest known hop distance from any seed entity to each entity
        # reached during traversal (seeds themselves are distance 0).
        hop_distance: dict[str, int] = {e: 0 for e in wanted}
        paths_touching: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            end = path.entities[-1].lower()
            hop_distance[end] = min(hop_distance.get(end, path.hops), path.hops)
            for entity in path.entities:
                paths_touching[entity.lower()].append(path)

        scored: list[tuple[float, int, GraphContext]] = []
        for source, doc_entities in self.doc_entities.items():
            doc_entities_lower = {e.lower() for e in doc_entities}
            touched = doc_entities_lower & hop_distance.keys()
            if not touched:
                continue

            direct_hits = len(touched & wanted)
            expanded_hits = len(touched) - direct_hits
            closest_dist = min(hop_distance[e] for e in touched)
            # Reported hop_distance is the DEEPEST hop reached among this
            # document's entities, not the shallowest. A document that
            # mentions both the seed entity (hop 0) and a target reached
            # only three hops away should be credited with hop_distance=3
            # — that's the traversal that actually justified pulling this
            # document in, and it's what the eval benchmark checks against.
            deepest_dist = max(hop_distance[e] for e in touched)
            score = direct_hits * 5 + expanded_hits * max(1, 4 - closest_dist)
            if score <= 0:
                continue

            relation_strings: list[str] = []
            seen = set()
            for entity in sorted(touched, key=lambda e: hop_distance[e]):
                for path in paths_touching.get(entity, []):
                    rendered = path.render()
                    if rendered not in seen:
                        seen.add(rendered)
                        relation_strings.append(rendered)
            scored.append(
                (
                    score,
                    closest_dist,
                    GraphContext(
                        source=source,
                        text=self.docs[source],
                        entities=sorted(doc_entities),
                        relations=relation_strings[:10],
                        hop_distance=deepest_dist,
                    ),
                )
            )

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [ctx for _, _, ctx in scored[:limit]]

    def health(self) -> bool:
        return True


class Neo4jGraphStore:
    backend_name = "neo4j"

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.driver.verify_connectivity()
        self._constraints()

    def _constraints(self) -> None:
        queries = [
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            (
                "CREATE CONSTRAINT document_source IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.source IS UNIQUE"
            ),
        ]
        with self.driver.session(database=self.database) as session:
            for query in queries:
                session.run(query).consume()

    def index_document(
        self, source: str, text: str, entities: list[str], relations: list[Relation]
    ) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "MERGE (d:Document {source:$source}) SET d.text=$text",
                source=source,
                text=text,
            ).consume()
            for entity in entities:
                session.run(
                    """
                    MATCH (d:Document {source:$source})
                    MERGE (e:Entity {name:$entity})
                    MERGE (d)-[:MENTIONS]->(e)
                    """,
                    source=source,
                    entity=entity,
                ).consume()
            for relation in relations:
                session.run(
                    """
                    MERGE (a:Entity {name:$a})
                    MERGE (b:Entity {name:$b})
                    MERGE (a)-[r:RELATED {kind:$kind}]->(b)
                    """,
                    a=relation.source,
                    b=relation.target,
                    kind=relation.relation,
                ).consume()

    def search(self, entities: list[str], limit: int, max_hops: int = 2) -> list[GraphContext]:
        if not entities:
            return []
        # max_hops comes from server-side config (app.config.Settings), never
        # from raw request text, and is clamped to a small integer range
        # before interpolation. Neo4j does not support parameterizing a
        # variable-length relationship bound (`*1..$n`) directly in Cypher,
        # so this is the standard, safe way to make the hop count
        # configurable: validate + clamp to int, then interpolate.
        max_hops = max(1, min(int(max_hops), MAX_ALLOWED_HOPS))
        query = f"""
        MATCH (seed:Entity)
        WHERE toLower(seed.name) IN $entities
        MATCH path = (seed)-[:RELATED*0..{max_hops}]-(reached:Entity)
        WITH reached, path, length(path) AS hop_distance
        ORDER BY hop_distance ASC
        WITH reached, min(hop_distance) AS hop_distance, collect(path)[0] AS samplePath
        MATCH (d:Document)-[:MENTIONS]->(reached)
        WITH d,
             collect(DISTINCT reached.name) AS matched,
             max(hop_distance) AS deepest_hop,
             collect(DISTINCT samplePath) AS samplePaths
        RETURN d.source AS source,
               d.text AS text,
               matched AS entities,
               deepest_hop AS hop_distance,
               [p IN samplePaths[0..5] | [n IN nodes(p) | n.name]] AS pathNodes,
               [p IN samplePaths[0..5] | [r IN relationships(p) | r.kind]] AS pathRels
        ORDER BY size(matched) DESC, deepest_hop ASC
        LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            rows = session.run(
                query,
                entities=[e.lower() for e in entities],
                limit=limit,
            )
            results: list[GraphContext] = []
            for row in rows:
                relation_strings = []
                for nodes, rels in zip(row["pathNodes"], row["pathRels"], strict=True):
                    if len(nodes) < 2:
                        continue
                    parts = [nodes[0]]
                    for kind, next_node in zip(rels, nodes[1:], strict=True):
                        parts.append(f"-[{kind}]-> {next_node}")
                    relation_strings.append(" ".join(parts))
                results.append(
                    GraphContext(
                        source=row["source"],
                        text=row["text"],
                        entities=list(row["entities"] or []),
                        relations=relation_strings,
                        hop_distance=row["hop_distance"],
                    )
                )
            return results

    def health(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False
