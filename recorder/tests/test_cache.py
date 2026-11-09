"""Unit tests for recorder/cache.py (M2-T6). No live API calls."""

import shutil

import pytest

from recorder.cache import ToolResultCache, disable_llm_cache, enable_llm_cache, CACHE_DIR

TEST_CACHE_DIR = CACHE_DIR / "test-m2-t6"


@pytest.fixture(autouse=True)
def _clean_cache_dir():
    yield
    shutil.rmtree(TEST_CACHE_DIR, ignore_errors=True)


def test_tool_result_cache_set_then_get():
    cache = ToolResultCache("example", cache_dir=TEST_CACHE_DIR)
    assert cache.get("web_search", {"query": "x"}) is None

    cache.set("web_search", {"query": "x"}, "the answer")
    assert cache.get("web_search", {"query": "x"}) == "the answer"


def test_tool_result_cache_different_args_miss():
    cache = ToolResultCache("example", cache_dir=TEST_CACHE_DIR)
    cache.set("web_search", {"query": "x"}, "the answer")

    assert cache.get("web_search", {"query": "y"}) is None


def test_tool_result_cache_key_order_independent():
    cache = ToolResultCache("example", cache_dir=TEST_CACHE_DIR)
    cache.set("calculator", {"a": 1, "b": 2}, "3")

    assert cache.get("calculator", {"b": 2, "a": 1}) == "3"


def test_tool_result_cache_persists_across_instances():
    cache = ToolResultCache("example", cache_dir=TEST_CACHE_DIR)
    cache.set("web_search", {"query": "x"}, "persisted")

    reloaded = ToolResultCache("example", cache_dir=TEST_CACHE_DIR)
    assert reloaded.get("web_search", {"query": "x"}) == "persisted"


def test_enable_llm_cache_sets_a_global_cache_without_live_call():
    from langchain_core.globals import get_llm_cache

    enable_llm_cache("example", cache_dir=TEST_CACHE_DIR)
    assert get_llm_cache() is not None


def test_disable_llm_cache_clears_the_global_cache():
    from langchain_core.globals import get_llm_cache

    enable_llm_cache("example", cache_dir=TEST_CACHE_DIR)
    assert get_llm_cache() is not None

    disable_llm_cache()
    assert get_llm_cache() is None
