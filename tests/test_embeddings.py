from app.embeddings import HashingEmbedder, cosine_similarity


def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder(64)
    a = embedder.embed("GPU cluster monitoring")
    b = embedder.embed("GPU cluster monitoring")
    assert a == b
    assert len(a) == 64


def test_related_text_has_positive_similarity():
    embedder = HashingEmbedder(128)
    a = embedder.embed("Qdrant vector search embeddings")
    b = embedder.embed("Qdrant stores vector embeddings")
    assert cosine_similarity(a, b) > 0
