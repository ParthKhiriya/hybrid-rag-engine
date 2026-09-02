import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.embeddings import Embedder
from app.graph_store import GraphStore, Relation
from app.models import DocumentIn, IngestResponse
from app.security import classify_trust, scan_text
from app.vector_store import VectorRecord, VectorStore

RELATION_PATTERNS = {
    "DEPENDS_ON": re.compile(
        r"([A-Z][A-Za-z0-9 _-]{2,40}) depends on ([A-Z][A-Za-z0-9 _-]{2,40})", re.I
    ),
    "MONITORED_BY": re.compile(
        r"([A-Z][A-Za-z0-9 _-]{2,40}) is monitored by ([A-Z][A-Za-z0-9 _-]{2,40})",
        re.I,
    ),
    "RUNS_ON": re.compile(
        r"([A-Z][A-Za-z0-9 _-]{2,40}) runs on ([A-Z][A-Za-z0-9 _-]{2,40})", re.I
    ),
    "OWNED_BY": re.compile(
        r"([A-Z][A-Za-z0-9 _-]{2,40}) is owned by ([A-Z][A-Za-z0-9 _-]{2,40})", re.I
    ),
}

ENTITY_PATTERN = re.compile(r"\b(?:[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){0,3})\b")


@dataclass(slots=True)
class Chunk:
    id: str
    source: str
    text: str
    metadata: dict[str, str]


def chunk_text(source: str, text: str, metadata: dict[str, str], size: int = 700) -> list[Chunk]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > size:
            chunks.append(Chunk(str(uuid.uuid4()), source, current, metadata))
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(Chunk(str(uuid.uuid4()), source, current, metadata))
    return chunks


def normalize_entity(value: str) -> str:
    value = value.strip(" .,:;()")
    for prefix in ("The ", "the ", "A ", "An ", "a ", "an "):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    return value.strip()


def extract_entities(text: str) -> list[str]:
    stop = {
        "The", "This", "A", "An", "It", "For", "When", "Each", "Production",
        "What", "Which", "How", "Where", "Who", "Why",
    }
    entities = {normalize_entity(m.group(0)) for m in ENTITY_PATTERN.finditer(text)}
    return sorted(e for e in entities if len(e) > 2 and e not in stop)[:80]


def extract_relations(text: str) -> list[Relation]:
    relations: list[Relation] = []
    for kind, pattern in RELATION_PATTERNS.items():
        for match in pattern.finditer(text):
            relations.append(
                Relation(
                    source=normalize_entity(match.group(1)),
                    relation=kind,
                    target=normalize_entity(match.group(2)),
                )
            )
    return relations


class IngestionService:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        graph_store: GraphStore,
        block_quarantined: bool = False,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.block_quarantined = block_quarantined

    def ingest(self, documents: list[DocumentIn]) -> IngestResponse:
        chunk_count = entity_count = relation_count = 0
        quarantined: list[str] = []
        for doc in documents:
            scan = scan_text(doc.text)
            trust = classify_trust(doc.trust, scan)
            if trust == "quarantined":
                quarantined.append(doc.source)
                if self.block_quarantined:
                    # Reject outright: don't index content that looks like
                    # an injection attempt at all.
                    continue

            entities = extract_entities(doc.text)
            relations = extract_relations(doc.text)
            entity_count += len(entities)
            relation_count += len(relations)
            self.graph_store.index_document(doc.source, doc.text, entities, relations)

            metadata = {**doc.metadata, "trust": trust}
            records = []
            for chunk in chunk_text(doc.source, doc.text, metadata):
                records.append(
                    VectorRecord(
                        id=chunk.id,
                        vector=self.embedder.embed(chunk.text),
                        text=chunk.text,
                        source=chunk.source,
                        metadata=chunk.metadata,
                    )
                )
            self.vector_store.upsert(records)
            chunk_count += len(records)

        return IngestResponse(
            documents=len(documents),
            chunks=chunk_count,
            entities=entity_count,
            relations=relation_count,
            backends={
                "vector": self.vector_store.backend_name,
                "graph": self.graph_store.backend_name,
            },
            quarantined=quarantined,
        )


def load_sample_documents() -> list[DocumentIn]:
    base = Path(__file__).resolve().parent.parent / "sample_data"
    docs = []
    for path in sorted(base.glob("*.md")):
        docs.append(
            DocumentIn(
                source=path.name,
                text=path.read_text(),
                metadata={"kind": "sample"},
                trust="trusted",
            )
        )
    return docs
