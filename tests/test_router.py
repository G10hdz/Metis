"""Tests for the router node — validates dict return and classification logic."""

import json
import pytest
from src.graph.state import MetisState
from src.graph.nodes import router, ROUTE_RAG, ROUTE_CODE, ROUTE_GENERAL, ROUTE_SEARCH, ROUTE_SEARCH_CONTINUE


def _make_state(query: str, search_context: str = "") -> MetisState:
    return MetisState(query=query, search_context=search_context)


class TestRouterDictReturn:
    """Ensure router always returns a dict compatible with StateGraph."""

    def test_returns_dict(self):
        state = _make_state("hello world")
        result = router(state)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        state = _make_state("hello world")
        result = router(state)
        assert "route" in result
        assert "next_node" in result

    def test_empty_query_returns_formatter(self):
        state = _make_state("")
        result = router(state)
        assert result["next_node"] == "formatter"
        assert result["response"] == "Empty query."
        assert result["route"] == ROUTE_GENERAL


class TestRouterClassification:
    """Validate keyword-based routing."""

    @pytest.mark.parametrize("query", [
        "write a python function to sort a list",
        "how to debug this error",
        "create a docker compose file",
        "implement an API endpoint in FastAPI",
    ])
    def test_code_route(self, query):
        state = _make_state(query)
        result = router(state)
        assert result["route"] == ROUTE_CODE
        assert result["next_node"] == ROUTE_CODE

    @pytest.mark.parametrize("query", [
        "what is machine learning",
        "explain the concept of gravity",
        "how to bake a cake",
        "describe the history of Python",
    ])
    def test_rag_route(self, query):
        state = _make_state(query)
        result = router(state)
        assert result["route"] == ROUTE_RAG
        assert result["next_node"] == ROUTE_RAG

    @pytest.mark.parametrize("query", [
        "hello",
        "good morning",
        "thank you",
        "42",
    ])
    def test_general_route(self, query):
        state = _make_state(query)
        result = router(state)
        assert result["route"] == ROUTE_GENERAL
        assert result["next_node"] == ROUTE_GENERAL


class TestRouterStateMerge:
    """Verify router output can merge into MetisState."""

    def test_merge_into_state(self):
        state = _make_state("write python code")
        result = router(state)
        # Simulate LangGraph merge: update state with result dict
        for key, value in result.items():
            setattr(state, key, value)
        assert state.route == ROUTE_CODE
        assert state.next_node == ROUTE_CODE


class TestSearchRoute:
    """Validate search keyword routing."""

    @pytest.mark.parametrize("query", [
        "latest news on AI",
        "search for recent updates on quantum computing",
        "what's new in Rust 2025",
        "find trending topics in tech 2026",
        "breaking news about AWS",
    ])
    def test_search_route(self, query):
        state = _make_state(query)
        result = router(state)
        assert result["route"] == ROUTE_SEARCH
        assert result["next_node"] == ROUTE_SEARCH

    @pytest.mark.parametrize("query,ctx", [
        ("go deeper", '{"learnings":["x"],"sources":[]}'),
        ("deeper", '{"learnings":["x"],"sources":[]}'),
        ("tell me more", '{"learnings":["x"],"sources":[]}'),
        ("dig deeper", '{"learnings":["x"],"sources":[]}'),
    ])
    def test_search_continue_route(self, query, ctx):
        state = _make_state(query, search_context=ctx)
        result = router(state)
        assert result["route"] == ROUTE_SEARCH_CONTINUE
        assert result["next_node"] == ROUTE_SEARCH_CONTINUE

    def test_deeper_without_context_falls_to_general(self):
        """If user says 'deeper' but there's no search_context, should go general."""
        state = _make_state("deeper", search_context="")
        result = router(state)
        assert result["route"] == ROUTE_GENERAL
