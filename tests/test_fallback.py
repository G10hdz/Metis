"""Tests for the fallback chain — timeout detection, VRAM parsing, tier return format."""

import pytest
import subprocess
import json

from src.utils.fallback import (
    _is_vram_error,
    VRAMError,
    TierTimeout,
    call_with_fallback,
    TIER_OLLAMA,
    TIER_OPENCODE,
    TIER_QWEN,
    TIER_GEMINI,
    TIER_TELEGRAM,
)
from src.graph.state import MetisState
from src.graph.nodes import router, ROUTE_SEARCH, ROUTE_SEARCH_CONTINUE, ROUTE_GENERAL


class TestVRAMErrorDetection:
    """Verify VRAM error string matching."""

    @pytest.mark.parametrize("msg", [
        "CUDA out of memory: tried to allocate 2.5 GiB",
        "Error: ROCm memory pool exhausted",
        "failed to allocate memory for tensor",
        "RuntimeError: CUDA OOM",
        "Memory allocation failed: not enough memory",
    ])
    def test_vram_error_detected(self, msg):
        assert _is_vram_error(Exception(msg)) is True

    @pytest.mark.parametrize("msg", [
        "Connection refused",
        "Model not found",
        "Timeout after 30s",
        "HTTP 500",
    ])
    def test_non_vram_error_not_detected(self, msg):
        assert _is_vram_error(Exception(msg)) is False


class TestFallbackReturnFormat:
    """Ensure call_with_fallback always returns a dict with response + tier."""

    @pytest.mark.skipif(
        True,  # Skip by default — requires live tools. Run manually with `pytest -k skip=false`
        reason="Requires live Ollama/opencode/qwen. Run manually."
    )
    def test_returns_dict_on_ollama_failure(self):
        """When Ollama fails (no server running), should cascade through tiers."""
        result = call_with_fallback("test query", task_type="general")
        assert isinstance(result, dict)
        assert "response" in result
        assert "tier" in result
        assert result["tier"] in (TIER_OLLAMA, TIER_OPENCODE, TIER_QWEN, TIER_GEMINI, TIER_TELEGRAM, "none")

    @pytest.mark.skipif(True, reason="Requires live tools. Run manually.")
    def test_response_is_string(self):
        result = call_with_fallback("hello", task_type="general")
        assert isinstance(result["response"], str)


class TestTierConstants:
    """Verify tier label constants."""

    def test_tier_labels(self):
        assert TIER_OLLAMA == "ollama"
        assert TIER_OPENCODE == "opencode"
        assert TIER_QWEN == "qwen"
        assert TIER_GEMINI == "gemini"
        assert TIER_TELEGRAM == "telegram"


class TestFallbackSubprocess:
    """Test subprocess-based tiers (skipped — tools may hang in CI)."""

    @pytest.mark.skipif(True, reason="opencode may hang. Run manually.")
    def test_opencode_not_found_returns_empty(self):
        """If opencode binary is not in PATH, should return empty string."""
        from src.utils.fallback import _tier_opencode
        try:
            result = _tier_opencode("test", timeout_s=5)
            assert isinstance(result, str)
        except FileNotFoundError:
            pytest.skip("opencode not installed")

    @pytest.mark.skipif(True, reason="qwen may hang. Run manually.")
    def test_qwen_not_found_returns_empty(self):
        """If qwen binary is not in PATH, should return empty string."""
        from src.utils.fallback import _tier_qwen
        try:
            result = _tier_qwen("test", timeout_s=5)
            assert isinstance(result, str)
        except FileNotFoundError:
            pytest.skip("qwen CLI not installed")


class TestSearchRouteWithFallback:
    """Search route classification still works after nodes update."""

    def _make_state(self, query: str, search_context: str = "") -> MetisState:
        return MetisState(query=query, search_context=search_context)

    @pytest.mark.parametrize("query", [
        "latest news on AI",
        "search for recent updates on quantum computing",
    ])
    def test_search_route(self, query):
        state = self._make_state(query)
        result = router(state)
        assert result["route"] == ROUTE_SEARCH

    def test_deeper_without_context_falls_to_general(self):
        state = self._make_state("deeper", search_context="")
        result = router(state)
        assert result["route"] == ROUTE_GENERAL
