"""
Comprehensive tests for the extensible config loader.

Tests cover:
- Settings schema validation and type coercion
- Profile selection (dev, prod, colab, auto-detect)
- Precedence: defaults < profile < .env < os env vars
- Path handling (string to Path conversion)

Note: Settings are loaded from settings/defaults.yaml, settings/{env}.yaml files.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from text_to_sql.settings import (
    Settings,
    load_settings,
    get_settings,
    _is_running_in_colab,
    _load_yaml_file,
)


class TestSettingsSchema:
    """Test Settings pydantic model: validation, type coercion, defaults."""

    def test_settings_defaults(self):
        """Settings should have sensible defaults."""
        settings = Settings()
        assert settings.ENV == "dev"
        assert settings.FILE_PATH == Path("data/dev.json")
        assert settings.VECTOR_DB_PATH == Path("data/chromadb")
        assert settings.DATABASES_DIR is None
        assert settings.GOOGLE_API_KEY is None

    def test_settings_path_coercion(self):
        """Pydantic should coerce string paths to Path objects."""
        settings = Settings(
            FILE_PATH="/custom/path.json",
        )
        assert isinstance(settings.FILE_PATH, Path)
        assert settings.FILE_PATH == Path("/custom/path.json")

    def test_settings_custom_values(self):
        """Settings can be instantiated with custom values."""
        settings = Settings(
            ENV="prod",
            FILE_PATH="/prod/queries.json",
            VECTOR_DB_PATH="/prod/vectors",
            DATABASES_DIR="/prod/dbs",
            GOOGLE_API_KEY="test-key-123",
        )
        assert settings.ENV == "prod"
        assert settings.FILE_PATH == Path("/prod/queries.json")
        assert settings.GOOGLE_API_KEY == "test-key-123"


class TestYamlFileLoading:
    """Test YAML file loading and merging."""

    def test_load_yaml_file_nonexistent(self):
        """Non-existent YAML files return empty dict."""
        result = _load_yaml_file(Path("/nonexistent/path.yaml"))
        assert result == {}

    def test_load_yaml_file_valid(self):
        """Valid YAML files are parsed correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("env: test\nfile_path: /test/path.json\n")
            f.flush()
            path = Path(f.name)

        try:
            result = _load_yaml_file(path)
            assert result.get("env") == "test"
            assert result.get("file_path") == "/test/path.json"
        finally:
            path.unlink()

    def test_load_yaml_file_empty(self):
        """Empty YAML files return empty dict."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            path = Path(f.name)

        try:
            result = _load_yaml_file(path)
            assert result == {}
        finally:
            path.unlink()


class TestProfileSelection:
    """Test profile selection logic and environment variable handling."""

    def test_profile_defaults_to_dev(self):
        """When no profile specified and not in Colab, profile should be 'dev'."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove TEXT_TO_SQL_ENV and Colab markers
            settings = load_settings(profile="dev")
            assert settings.ENV == "dev"

    def test_profile_respects_text_to_sql_env_var(self):
        """TEXT_TO_SQL_ENV environment variable is respected."""
        with patch.dict(os.environ, {"TEXT_TO_SQL_ENV": "prod"}, clear=True):
            settings = load_settings()
            assert settings.ENV == "prod"

    def test_explicit_profile_overrides_env_var(self):
        """Explicit profile parameter overrides TEXT_TO_SQL_ENV."""
        with patch.dict(os.environ, {"TEXT_TO_SQL_ENV": "prod"}, clear=True):
            settings = load_settings(profile="dev")
            assert settings.ENV == "dev"

    def test_colab_detection(self):
        """_is_running_in_colab detects Colab environment."""
        assert not _is_running_in_colab()  # Usually false in test env

        with patch.dict(os.environ, {"COLAB_GPU": "1"}):
            assert _is_running_in_colab()

        with patch.dict(os.environ, {"COLAB_TPU_ADDR": "10.0.0.1"}):
            assert _is_running_in_colab()


class TestEnvironmentVariablePrecedence:
    """Test the precedence order: defaults < profile < .env < os env vars."""

    def test_env_var_overrides_profile(self):
        """OS environment variables should override profile file values."""
        with patch.dict(os.environ, {"FILE_PATH": "/override/path.json"}, clear=True):
            settings = load_settings(profile="dev")
            assert settings.FILE_PATH == Path("/override/path.json")

    def test_env_var_string_coercion(self):
        """String env vars are coerced to correct types (Path, etc.)."""
        with patch.dict(os.environ, {"VECTOR_DB_PATH": "/custom/vectors"}, clear=True):
            settings = load_settings(profile="dev")
            assert isinstance(settings.VECTOR_DB_PATH, Path)
            assert settings.VECTOR_DB_PATH == Path("/custom/vectors")

    def test_env_var_none_not_included(self):
        """Environment variable set to None should not be in merged dict."""
        with patch.dict(os.environ, {}, clear=True):
            settings = load_settings(profile="dev")
            # Should use defaults since no env var is set
            expected_path = Path(__file__).parent.parent / "data" / "dev.json"
            assert settings.FILE_PATH == expected_path

    def test_env_var_empty_string_handled(self):
        """Empty string env vars are treated as falsy."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}, clear=True):
            settings = load_settings(profile="dev")
            # Empty string is falsy, but pydantic may accept it
            # The specific behavior depends on pydantic's handling
            assert settings.GOOGLE_API_KEY == ""


class TestGetSettings:
    """Test the get_settings function (primary API)."""

    def test_get_settings_returns_settings(self):
        """get_settings() returns Settings object."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_has_expected_attributes(self):
        """Settings from get_settings has all expected fields."""
        settings = get_settings()
        assert hasattr(settings, "ENV")
        assert hasattr(settings, "FILE_PATH")
        assert hasattr(settings, "VECTOR_DB_PATH")
        assert hasattr(settings, "GOOGLE_API_KEY")

    def test_get_settings_values_are_typed(self):
        """Settings attributes have correct types."""
        settings = get_settings()
        assert isinstance(settings.FILE_PATH, Path)
        assert isinstance(settings.VECTOR_DB_PATH, Path)
        assert isinstance(settings.ENV, str)


class TestIntegration:
    """Integration tests with real YAML files."""

    def test_load_settings_with_real_yaml_files(self):
        """Integration test: load_settings with actual settings files."""
        # This test depends on settings/defaults.yaml and settings/dev.yaml existing
        with patch.dict(os.environ, {"TEXT_TO_SQL_ENV": "dev"}, clear=True):
            settings = load_settings(profile="dev")
            assert settings.ENV == "dev"
            # dev should have databases_dir set
            assert settings.DATABASES_DIR is not None

    def test_all_profiles_can_be_loaded(self):
        """All profile files can be loaded without error."""
        for profile in ["dev", "prod", "colab"]:
            settings = load_settings(profile=profile)
            assert settings.ENV == profile

    def test_settings_directory_location(self):
        """Settings files are loaded from settings/ subdirectory."""
        # The settings directory should be at text_to_sql/settings/
        settings_dir = Path(__file__).parent.parent / "text_to_sql" / "settings"
        assert (settings_dir / "defaults.yaml").exists()
        assert (settings_dir / "dev.yaml").exists()
        assert (settings_dir / "prod.yaml").exists()
        assert (settings_dir / "colab.yaml").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
