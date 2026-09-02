from app.graph_store import InMemoryGraphStore, Relation


def build_chain_store() -> InMemoryGraphStore:
    """A -DEPENDS_ON-> B -RUNS_ON-> C -MONITORED_BY-> D, all mentioned in
    one document, plus an unrelated document with no path to the chain."""
    store = InMemoryGraphStore()
    store.index_document(
        "chain.md",
        "A depends on B. B runs on C. C is monitored by D.",
        ["A", "B", "C", "D"],
        [
            Relation(source="A", relation="DEPENDS_ON", target="B"),
            Relation(source="B", relation="RUNS_ON", target="C"),
            Relation(source="C", relation="MONITORED_BY", target="D"),
        ],
    )
    store.index_document(
        "unrelated.md",
        "Z depends on Y.",
        ["Z", "Y"],
        [Relation(source="Z", relation="DEPENDS_ON", target="Y")],
    )
    return store


def test_direct_neighbor_found_at_one_hop():
    store = build_chain_store()
    results = store.search(["A"], limit=5, max_hops=1)
    assert results
    assert results[0].source == "chain.md"
    assert results[0].hop_distance == 1  # B is 1 hop from A; D is out of range


def test_two_hop_target_reached_within_budget():
    store = build_chain_store()
    results = store.search(["A"], limit=5, max_hops=2)
    assert results[0].hop_distance == 2  # reaches C, not yet D


def test_three_hop_target_reached_within_budget():
    store = build_chain_store()
    results = store.search(["A"], limit=5, max_hops=3)
    assert results[0].hop_distance == 3  # reaches D
    assert any(r.endswith("-> D") for r in results[0].relations)


def test_hop_budget_is_respected_not_exceeded():
    store = build_chain_store()
    results = store.search(["A"], limit=5, max_hops=1)
    # D is 3 hops away; with a budget of 1 it must not appear as a
    # traversed entity in any rendered path (checked as an exact hop
    # target, not a raw substring, since "DEPENDS_ON" also contains "D").
    assert not any(r.endswith("-> D") or " D " in r for r in results[0].relations)
    assert results[0].hop_distance == 1


def test_traversal_is_undirected_reverse_query_works():
    store = build_chain_store()
    # Asking from D's side should still find the chain back to A.
    results = store.search(["D"], limit=5, max_hops=3)
    assert results
    assert results[0].source == "chain.md"
    assert results[0].hop_distance == 3


def test_unrelated_documents_are_not_pulled_in():
    store = build_chain_store()
    results = store.search(["A"], limit=5, max_hops=3)
    sources = {r.source for r in results}
    assert "unrelated.md" not in sources


def test_max_hops_is_clamped_to_hard_ceiling():
    store = build_chain_store()
    # Requesting an absurd hop budget must not error or traverse unbounded.
    results = store.search(["A"], limit=5, max_hops=999)
    assert results[0].hop_distance == 3  # still just reaches D, the true diameter


def test_no_entities_returns_empty():
    store = build_chain_store()
    assert store.search([], limit=5, max_hops=2) == []
