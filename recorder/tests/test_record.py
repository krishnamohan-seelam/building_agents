"""Unit tests for recorder/record.py (M2-T8). LLM mocked throughout; no live API calls."""

import shutil
from unittest.mock import patch

import pytest
from jsonschema import Draft202012Validator
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from recorder.record import record_run, REPO_ROOT, SCHEMA_PATH

TRACES_DIR = REPO_ROOT / "traces" / "query-decomposer"


class _CountingFakeChatModel(FakeListChatModel):
    """A real BaseChatModel (not a MagicMock) so LangGraph's token streaming and LangChain's
    LLM cache both work, with a plain call counter (pydantic models reject ad-hoc
    monkeypatching of their methods, so this subclasses instead of patching an instance)."""

    call_count: int = 0

    def _generate(self, *args, **kwargs):
        self.call_count += 1
        return super()._generate(*args, **kwargs)

    def _stream(self, *args, **kwargs):
        self.call_count += 1
        yield from super()._stream(*args, **kwargs)


@pytest.fixture(autouse=True)
def _clean_traces_dir():
    yield
    shutil.rmtree(TRACES_DIR, ignore_errors=True)
    shutil.rmtree(REPO_ROOT / "recorder" / ".cache", ignore_errors=True)


@pytest.fixture
def fake_llm():
    llm = _CountingFakeChatModel(responses=["Final Answer: 42"])
    with patch("query_decomposer.agent.get_llm", return_value=llm):
        yield llm


def test_record_run_writes_a_schema_valid_trace(fake_llm):
    trace = record_run("query_decomposer", input_index=1, cached=False)

    import json

    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    Draft202012Validator(schema).validate(trace)

    out_path = TRACES_DIR / "query_decomposer-input1.json"
    assert out_path.exists()


def test_record_run_rejects_out_of_range_input_index(fake_llm):
    with pytest.raises(ValueError):
        record_run("query_decomposer", input_index=99, cached=False)


def test_cached_rerun_makes_zero_additional_model_calls(fake_llm):
    record_run("query_decomposer", input_index=1, cached=True)
    calls_after_first = fake_llm.call_count
    assert calls_after_first >= 1

    record_run("query_decomposer", input_index=1, cached=True)
    assert fake_llm.call_count == calls_after_first


def test_uncached_run_after_a_cached_run_is_not_served_from_the_global_cache(fake_llm):
    """set_llm_cache is process-global; a cached=False run must not inherit a prior
    cached=True run's cache (regression: this used to leak across calls)."""
    record_run("query_decomposer", input_index=1, cached=True)
    calls_after_cached = fake_llm.call_count

    record_run("query_decomposer", input_index=1, cached=False)
    assert fake_llm.call_count > calls_after_cached


def test_recorded_trace_includes_tool_call_and_tool_result_steps_for_a_research_cycle():
    llm = FakeListChatModel(
        responses=["Follow-up [web_search]: who founded Apple?", "Final Answer: Steve Jobs"]
    )
    mock_tool = type("MockTool", (), {"invoke": staticmethod(lambda args: "Steve Jobs and others")})()

    with patch("query_decomposer.agent.get_llm", return_value=llm), \
         patch.dict("query_decomposer.agent.TOOLS", {"web_search": mock_tool}):
        trace = record_run("query_decomposer", input_index=1, cached=False)

    kinds = [s["kind"] for s in trace["steps"]]
    assert "tool_call" in kinds
    assert "tool_result" in kinds

    tool_result_step = next(s for s in trace["steps"] if s["kind"] == "tool_result")
    assert tool_result_step["detail"]["output"] == "Steve Jobs and others"
