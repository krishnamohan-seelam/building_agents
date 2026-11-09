"""Model + tool-result caching for `record --cached` (M2-T6).

Storage lives under recorder/.cache/ (git-ignored) and never ships in a published trace.
"""

import json
from pathlib import Path
from typing import Any

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def enable_llm_cache(example: str, cache_dir: Path = CACHE_DIR) -> None:
    """Points LangChain's global LLM cache at a per-example SQLite file.

    A changed prompt misses this cache (and every call downstream of it), since the cache
    key is derived from the prompt content itself.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(cache_dir / f"{example}.sqlite3")))


def _cache_key(tool_name: str, args: dict) -> str:
    return json.dumps([tool_name, sorted(args.items())], sort_keys=True)


class ToolResultCache:
    """Caches tool results by (tool_name, args), backed by a JSON file."""

    def __init__(self, example: str, cache_dir: Path = CACHE_DIR):
        self._path = cache_dir / f"{example}-tools.json"
        cache_dir.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                self._store: dict[str, Any] = json.load(f)
        else:
            self._store = {}

    def get(self, tool_name: str, args: dict) -> Any | None:
        return self._store.get(_cache_key(tool_name, args))

    def set(self, tool_name: str, args: dict, result: Any) -> None:
        self._store[_cache_key(tool_name, args)] = result
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._store, f)
