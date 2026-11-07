import pytest

from ragagent.llm import parse_json
from ragagent.search import reciprocal_rank_fusion


def test_rrf_rewards_agreement():
    # doc 2 is ranked well by both lists, so it beats docs ranked first by only one list
    assert reciprocal_rank_fusion([[1, 2, 3], [4, 2, 5]])[0] == 2
    assert reciprocal_rank_fusion([[1, 2], [2, 1]], limit=1) == [1]   # tie -> lower id


@pytest.mark.parametrize("stage", ["bm25", "dense", "hybrid"])
def test_first_stages_find_statin_docs(engine, stage):
    top = engine.candidates("statins breast cancer", n=2, first_stage=stage)
    assert set(top) == {0, 1}


def test_search_returns_docs_and_coverage(engine):
    res = engine.search("statins and vitamin D", aspects=["statins breast cancer", "vitamin d bones"], top_k=3)
    assert len(res.docs) == 3
    assert res.coverage == 1.0
    assert res.uncovered_aspects == []


def test_aspect_reranker_adds_diversity(engine):
    aspects = ["statins breast cancer", "vitamin d bone loss"]
    ce = engine.search("statins breast cancer vitamin d", aspects=aspects, reranker="cross-encoder", top_k=2)
    aa = engine.search("statins breast cancer vitamin d", aspects=aspects, reranker="aspect", top_k=2)
    assert aa.coverage >= ce.coverage
    assert aa.coverage == 1.0


def test_exclude_and_invalid_args(engine):
    res = engine.search("statins breast cancer", exclude=[0, 1], top_k=2)
    assert {d.idx for d in res.docs}.isdisjoint({0, 1})
    with pytest.raises(ValueError):
        engine.search("x", reranker="magic")
    with pytest.raises(ValueError):
        engine.candidates("x", first_stage="magic")


def test_parse_json_tolerates_prose():
    assert parse_json('Sure! ```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_lean_bm25_matches_rank_bm25():
    import numpy as np
    from rank_bm25 import BM25Okapi

    from fakes import CORPUS
    from ragagent.core.retrievers import tokenize
    from ragagent.lean_bm25 import LeanBM25

    texts = [f"{t}. {x}" for t, x in CORPUS] * 3
    ref, lean = BM25Okapi([tokenize(t) for t in texts]), LeanBM25(texts)
    for q in ["statins breast cancer", "vitamin d d bones", "coffee liver enzymes", "nothing matches"]:
        assert np.allclose(ref.get_scores(tokenize(q)), lean.get_scores(tokenize(q)), atol=1e-6)
