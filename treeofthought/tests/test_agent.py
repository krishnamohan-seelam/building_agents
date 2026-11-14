import json
from unittest.mock import MagicMock, patch

from treeofthought.agent import (
    router,
    generator_node,
    evaluator_node,
    pruner_node,
    synthesizer_node,
    best_pruned_sibling,
    log_event,
    logger,
)
from treeofthought.utils.thought import Thought, format_path, new_id
from treeofthought.utils.schemas import GeneratorOutput, ThoughtCandidate, EvaluatorOutput, SynthesisOutput


def _thought(id_="root", parent_id=None, path=None, score=None, status="active") -> Thought:
    return Thought(
        id=id_, parent_id=parent_id, path=path or [], content="", score=score, status=status,
    )


# ── utils/thought.py ──────────────────────────────────────────────

def test_format_path_empty() -> None:
    assert format_path(_thought(path=[])) == ""


def test_format_path_multi_step() -> None:
    text = format_path(_thought(path=["First step", "Second step"]))
    assert "Step1: First step" in text
    assert "Step2: Second step" in text


def test_new_id_root() -> None:
    assert new_id(None, 1) == "root"


def test_new_id_nested() -> None:
    assert new_id("root.2", 3) == "root.2.3"


# ── router ─────────────────────────────────────────────────────────

def _router_state(score, depth=0, max_depth=3, total_thoughts=1):
    return {
        "scored_thoughts": [_thought(score=score)],
        "all_thoughts": [_thought()] * total_thoughts,
        "depth": depth,
        "max_depth": max_depth,
    }


def test_router_finishes_on_solved_threshold() -> None:
    assert router(_router_state(score=9.0)) == "finish"


def test_router_finishes_on_max_depth() -> None:
    assert router(_router_state(score=2.0, depth=3, max_depth=3)) == "finish"


def test_router_finishes_on_max_total_thoughts() -> None:
    assert router(_router_state(score=2.0, total_thoughts=6)) == "finish"


def test_router_prunes_otherwise() -> None:
    assert router(_router_state(score=2.0, depth=0, total_thoughts=1)) == "prune"


# ── pruner_node ────────────────────────────────────────────────────

def test_pruner_node_keeps_beam_width_survivors() -> None:
    scored = [_thought("a", score=9.0), _thought("b", score=7.0), _thought("c", score=5.0)]
    state = {
        "scored_thoughts": scored,
        "all_thoughts": scored,
        "beam_width": 2,
        "depth": 0,
        "best_score_so_far": 0.0,
    }
    result = pruner_node(state)
    assert {t["id"] for t in result["frontier"]} == {"a", "b"}
    assert scored[2]["status"] == "pruned"
    assert result["depth"] == 1


def test_pruner_node_reactivates_best_pruned_sibling_on_regression() -> None:
    prior_pruned = _thought("x", score=8.0, status="pruned")
    scored = [_thought("a", score=3.0), _thought("b", score=2.0)]
    state = {
        "scored_thoughts": scored,
        "all_thoughts": scored + [prior_pruned],
        "beam_width": 1,
        "depth": 1,
        "best_score_so_far": 8.0,
    }
    result = pruner_node(state)
    frontier_ids = {t["id"] for t in result["frontier"]}
    assert "x" in frontier_ids
    assert prior_pruned["status"] == "active"


def test_best_pruned_sibling_returns_none_when_empty() -> None:
    assert best_pruned_sibling([_thought("a", status="active")]) is None


# ── nodes (LLM mocked) ────────────────────────────────────────────

@patch("treeofthought.agent.get_llm")
def test_generator_node_logging_and_output(mock_get_llm) -> None:
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = GeneratorOutput(
        candidates=[ThoughtCandidate(thought="Step A"), ThoughtCandidate(thought="Step B")]
    )
    mock_get_llm.return_value = mock_llm

    state = {
        "problem": "Pick a slogan.",
        "frontier": [_thought()],
        "all_thoughts": [],
        "depth": 0,
    }

    with patch.object(logger, "info") as mock_info:
        result = generator_node(state)
        assert len(result["pending_evaluation"]) == 2
        assert result["all_thoughts"] == result["pending_evaluation"]

        log_types = [json.loads(call[0][0])["event_type"] for call in mock_info.call_args_list]
        assert log_types == ["node_start", "llm_start", "llm_end", "node_end"]

        llm_start_data = json.loads(mock_info.call_args_list[1][0][0])["data"]
        assert "messages" in llm_start_data


@patch("treeofthought.agent.get_llm")
def test_evaluator_node_marks_solved(mock_get_llm) -> None:
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = EvaluatorOutput(
        score=9.0, reason="Converges well."
    )
    mock_get_llm.return_value = mock_llm

    state = {"problem": "Pick a slogan.", "pending_evaluation": [_thought("root.1", path=["Step A"])]}

    with patch.object(logger, "info"):
        result = evaluator_node(state)
        assert result["scored_thoughts"][0]["score"] == 9.0
        assert result["scored_thoughts"][0]["status"] == "solved"


@patch("treeofthought.agent.get_llm")
def test_synthesizer_node_picks_best_scored_thought(mock_get_llm) -> None:
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = SynthesisOutput(
        final_answer="The winning answer."
    )
    mock_get_llm.return_value = mock_llm

    state = {
        "problem": "Pick a slogan.",
        "all_thoughts": [
            _thought("root.1", path=["Weak"], score=3.0),
            _thought("root.2", path=["Strong"], score=8.5),
        ],
    }

    with patch.object(logger, "info"):
        result = synthesizer_node(state)
        assert result["final_answer"] == "The winning answer."
