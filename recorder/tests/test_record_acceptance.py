"""M2 acceptance (M2-T10): a real recorded query_decomposer trace carries real narration,
not just the M1 hand-written fixture's. LLM mocked; no live API calls."""

import shutil
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from recorder.record import record_run, REPO_ROOT

TRACES_DIR = REPO_ROOT / "traces" / "query-decomposer"


@pytest.fixture(autouse=True)
def _clean_traces_dir():
    yield
    shutil.rmtree(TRACES_DIR, ignore_errors=True)


def test_recorded_trace_contains_real_custom_narration():
    llm = FakeListChatModel(responses=["Final Answer: 42"])
    with patch("query_decomposer.agent.get_llm", return_value=llm):
        trace = record_run("query_decomposer", input_index=1, cached=False)

    custom_steps = [s for s in trace["steps"] if s["kind"] == "custom"]
    assert len(custom_steps) >= 1
    assert all(isinstance(s.get("summary"), str) and s["summary"] for s in custom_steps)
