"""
Configuration loader for text_to_sql RAG system.

Use `get_settings()` to load environment-specific configuration:

    from text_to_sql.settings import get_settings
    settings = get_settings()
    print(settings.ENV)      # Current profile: "dev", "prod", "colab"

Configuration sources (in precedence order):
    1. settings/defaults.yaml (fallback)
    2. settings/{ENV}.yaml (e.g., settings/prod.yaml)
    3. .env file (python-dotenv)
    4. OS environment variables (highest priority)

Profile is selected via:
    - TEXT_TO_SQL_ENV environment variable, OR
    - Auto-detection of Colab (COLAB_GPU/COLAB_TPU_ADDR markers), OR
    - Default to "dev"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator


COLAB_ENV_MARKERS = ["COLAB_GPU", "COLAB_TPU_ADDR"]


def _is_running_in_colab() -> bool:
    """Detect if code is running in Google Colab environment."""
    return any(marker in os.environ for marker in COLAB_ENV_MARKERS)


# Path to the data directory, relative to project root
DATA_PATH = Path(__file__).parent.parent / "data"


class Settings(BaseModel):
    """Configuration schema for text_to_sql RAG system.

    All fields are uppercase to match environment variable naming conventions.
    Use pydantic for type validation and IDE autocompletion.

    Fields:
        ENV: Current profile name (dev/prod/colab)
        FILE_PATH: Path to training data JSON file
        VECTOR_DB_PATH: Path to ChromaDB vector store directory
        DATABASES_DIR: Path to directory containing test/sample databases
        GOOGLE_API_KEY: Gemini API key (optional, from secrets/env)
    """

    ENV: str = "dev"
    FILE_PATH: Path = Path("data/dev.json")
    VECTOR_DB_PATH: Path = Path("data/chromadb")
    DATABASES_DIR: Optional[Path] = None
    GOOGLE_API_KEY: Optional[str] = None

    @field_validator(
        "FILE_PATH", "VECTOR_DB_PATH", "DATABASES_DIR", mode="before"
    )
    @classmethod
    def resolve_path(cls, v):
        if isinstance(v, str) and v:
            if os.path.isabs(v):
                return Path(v)
            return DATA_PATH / v
        return v


_SETTINGS_CACHE: Optional[Settings] = None


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load YAML file safely; return empty dict if not found or on error."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def load_settings(
    profile: Optional[str] = None, dotenv_path: Path = Path(".env")
) -> Settings:
    """Load settings with multi-source precedence.

    Precedence order (lowest to highest):
        1. settings/defaults.yaml - fallback defaults
        2. settings/{ENV}.yaml - environment-specific file (e.g., settings/prod.yaml)
        3. .env file - local overrides via python-dotenv
        4. OS environment variables - highest priority

    Args:
        profile: Environment profile to use (dev/prod/colab). If None, uses TEXT_TO_SQL_ENV
                 env var; if unset, auto-detects Colab or defaults to "dev".
        dotenv_path: Path to .env file to load (default: ".env" in project root).

    Returns:
        Settings: Fully typed, validated configuration object.
    """
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is not None and profile is None:
        return _SETTINGS_CACHE

    settings_dir = Path(__file__).parent / "settings"

    # 1) Load defaults
    defaults = _load_yaml_file(settings_dir / "defaults.yaml")

    # 2) Determine profile to use
    profile = profile or os.environ.get("TEXT_TO_SQL_ENV")
    if profile is None:
        profile = "colab" if _is_running_in_colab() else "dev"

    # 3) Load profile-specific settings
    profile_settings = _load_yaml_file(settings_dir / f"{profile}.yaml")

    # 4) Merge: defaults + profile settings (profile overrides defaults)
    merged: Dict[str, Any] = {}
    merged.update({k.upper(): v for k, v in defaults.items()})
    merged.update({k.upper(): v for k, v in profile_settings.items()})

    # 5) Load .env file into environment (does not override existing env vars)
    load_dotenv(dotenv_path, override=False)

    # 6) Overlay OS environment variables (highest priority)
    for field in Settings.model_fields:
        env_val = os.environ.get(field)
        if env_val is not None:
            merged[field] = env_val

    # 7) Instantiate Settings (pydantic handles type coercion)
    settings = Settings(**merged)

    # Cache for future calls (unless explicit profile requested)
    if profile and os.environ.get("TEXT_TO_SQL_ENV") is None:
        _SETTINGS_CACHE = settings

    return settings


def get_settings() -> Settings:
    """Get the current settings (cached after first call).

    Recommended way to access configuration in application code:

        from text_to_sql.settings import get_settings
        settings = get_settings()
        api_key = settings.GOOGLE_API_KEY  # Optional[str]

    Returns:
        Settings: Current environment settings.
    """
    return load_settings()
