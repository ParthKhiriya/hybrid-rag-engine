from app.evaluation import groundedness_proxy, percentile


def test_groundedness_proxy():
    score = groundedness_proxy("Qdrant stores vectors", ["Qdrant stores vector embeddings"])
    assert score > 0.5


def test_percentile():
    assert percentile([10, 20, 30, 40], 0.95) == 40
