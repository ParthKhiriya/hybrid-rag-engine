from app.retriever import reciprocal_rank_fusion


def test_rrf_rewards_documents_seen_in_multiple_rankings():
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]
