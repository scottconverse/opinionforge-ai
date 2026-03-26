"""Unit tests for application configuration loading."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from opinionforge.config import Settings, get_settings


class TestSettings:
    def test_default_llm_provider(self) -> None:
        """Settings defaults LLM provider to 'anthropic'."""
        with patch.dict(os.environ, {}, clear=True):
            s = Settings()
            assert s.opinionforge_llm_provider == "anthropic"

    def test_default_search_provider(self) -> None:
        """Settings defaults search provider to 'tavily'."""
        with patch.dict(os.environ, {}, clear=True):
            s = Settings()
            assert s.opinionforge_search_provider == "tavily"

    def test_reads_from_environment(self) -> None:
        """Settings reads values from environment variables."""
        env = {
            "OPINIONFORGE_LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key-placeholder",
            "OPINIONFORGE_SEARCH_API_KEY": "test-search-key-placeholder",
        }
        with patch.dict(os.environ, env, clear=True):
            s = Settings()
            assert s.opinionforge_llm_provider == "openai"
            assert s.openai_api_key == "test-key-placeholder"
            assert s.opinionforge_search_api_key == "test-search-key-placeholder"

    def test_require_llm_api_key_exits_when_missing(self) -> None:
        """require_llm_api_key raises SystemExit(5) when key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            s = Settings()
            with pytest.raises(SystemExit) as exc_info:
                s.require_llm_api_key()
            assert exc_info.value.code == 5

    def test_require_llm_api_key_returns_key_when_present(self) -> None:
        """require_llm_api_key returns the key when it is set."""
        env = {"ANTHROPIC_API_KEY": "test-key-placeholder"}
        with patch.dict(os.environ, env, clear=True):
            s = Settings()
            assert s.require_llm_api_key() == "test-key-placeholder"

    def test_require_search_api_key_exits_when_missing(self) -> None:
        """require_search_api_key raises SystemExit(5) when key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            s = Settings()
            with pytest.raises(SystemExit) as exc_info:
                s.require_search_api_key()
            assert exc_info.value.code == 5

    def test_require_openai_key_when_provider_is_openai(self) -> None:
        """require_llm_api_key checks openai key when provider is openai."""
        env = {
            "OPINIONFORGE_LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key-placeholder",
        }
        with patch.dict(os.environ, env, clear=True):
            s = Settings()
            assert s.require_llm_api_key() == "test-key-placeholder"


class TestGetSettings:
    def test_returns_settings_instance(self) -> None:
        """get_settings returns a Settings instance."""
        get_settings.cache_clear()
        s = get_settings()
        assert isinstance(s, Settings)
