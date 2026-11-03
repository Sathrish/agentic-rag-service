from fakes import FakeLLM
from ragagent.agent import NO_EVIDENCE, RagAgent, cited_numbers, excerpt, normalise_answer, split_sentences


def test_direct_route_skips_retrieval(engine):
    llm = FakeLLM(route="direct")
    out = RagAgent(engine, llm).run("hello there")
    assert out["route"] == "direct"
    assert out["sources"] == [] and out["rounds"] == 0
    assert "nutrition" in out["answer"]
    assert [t["step"] for t in out["trace"]] == ["plan", "answer"]


def test_happy_path_answers_with_citations_and_verifies(engine):
    llm = FakeLLM(aspects=["statins breast cancer"])
    out = RagAgent(engine, llm).run("Do statins cause breast cancer?")
    assert out["rounds"] == 1                       # enough evidence: no rewrite
    assert out["sources"] and out["faithfulness"] == 1.0
    assert [t["step"] for t in out["trace"]] == ["plan", "retrieve", "grade", "answer", "verify"]
    assert out["ranking"][:2] == [0, 1]


def test_thin_evidence_triggers_one_bounded_rewrite(engine):
    llm = FakeLLM(aspects=["statins breast cancer", "vitamin d bones"], relevant=[1])
    agent = RagAgent(engine, llm, max_rewrites=1, min_docs=5, min_coverage=1.0)
    out = agent.run("statins and vitamin D")
    steps = [t["step"] for t in out["trace"]]
    assert steps.count("rewrite") == 1 and out["rounds"] == 2     # stops after the cap
    second = [t for t in out["trace"] if t["step"] == "retrieve"][1]
    first = [t for t in out["trace"] if t["step"] == "retrieve"][0]
    assert set(second["docs"]).isdisjoint(first["docs"])          # never re-shows a document


def test_no_relevant_documents_refuses_to_guess(engine):
    llm = FakeLLM(aspects=["quantum gravity"], relevant=[])
    out = RagAgent(engine, llm, max_rewrites=0).run("quantum gravity?")
    assert out["answer"] == NO_EVIDENCE and out["sources"] == []


def test_unsupported_sentences_lower_faithfulness(engine):
    llm = FakeLLM(aspects=["statins breast cancer"], unsupported=[2])
    out = RagAgent(engine, llm).run("Do statins cause breast cancer?")
    assert out["faithfulness"] == 0.5
    assert out["unsupported"] == ["They may improve survival [2]."]


def test_planner_failure_falls_back_to_rules(engine):
    class Broken(FakeLLM):
        def json(self, system, user):
            if "plan searches" in system:
                raise ValueError("bad json")
            return super().json(system, user)
    out = RagAgent(engine, Broken()).run("Do statins cause breast cancer?")
    assert out["route"] == "retrieve" and out["aspects"]
    assert "rule-based" in out["trace"][0]["detail"]


def test_search_only_stops_before_answering(engine):
    llm = FakeLLM(aspects=["statins breast cancer"])
    out = RagAgent(engine, llm).search_only("Do statins cause breast cancer?")
    assert out["answer"] == "" and out["ranking"][:2] == [0, 1]
    assert [t["step"] for t in out["trace"]] == ["plan", "retrieve", "grade"]
    assert llm.calls == 2                            # plan + grade


def test_single_pass_baseline(engine):
    llm = FakeLLM()
    out = RagAgent(engine, llm).run_single_pass("Do statins cause breast cancer?")
    assert out["llm_calls"] == 2                    # answer + verify only
    assert len(out["sources"]) == 5


def test_helpers():
    assert split_sentences("A is true [1]. B too [2]! C?") == ["A is true [1].", "B too [2]!", "C?"]
    assert cited_numbers("x [2][1] y [2]") == [1, 2]
    assert normalise_answer("breast‑cancer risk【3】【 4 】") == "breast-cancer risk[3][4]"
    text = "Background " + "x" * 2000 + " Conclusion: statins help."
    short = excerpt(text, 900)
    assert short.startswith("Background") and short.endswith("statins help.") and len(short) <= 905
