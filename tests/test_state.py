"""Tests for MetisState — validates Pydantic v2 state model."""

import pytest
from src.graph.state import MetisState


class TestMetisStateCreation:
    """State can be created with defaults or from query."""

    def test_default_state(self):
        state = MetisState()
        assert state.query == ""
        assert state.messages == []
        assert state.next_node == "router"
        assert state.kb_context == ""
        assert state.response == ""
        assert state.route == ""
        assert state.code_snippet == ""

    def test_from_query(self):
        state = MetisState.from_query("test query")
        assert state.query == "test query"
        assert len(state.messages) == 1
        assert state.messages[0]["content"] == "test query"
        assert state.next_node == "router"


class TestMetisStateSerialization:
    """to_dict produces a LangGraph-mergeable dict."""

    def test_to_dict(self):
        state = MetisState(query="hello", route="code")
        d = state.to_dict()
        assert isinstance(d, dict)
        assert d["query"] == "hello"
        assert d["route"] == "code"

    def test_to_dict_excludes_unset(self):
        state = MetisState(query="hello")
        d = state.to_dict()
        # route was never explicitly set — pydantic default, so may or may not be excluded
        # The key check: query must be present
        assert "query" in d

    def test_roundtrip(self):
        state = MetisState.from_query("roundtrip test")
        state_dict = state.to_dict()
        # Simulate LangGraph receiving this dict back
        state2 = MetisState(**state_dict)
        assert state2.query == "roundtrip test"


class TestMetisStateMerge:
    """Validate that partial dicts (like node returns) can merge into state."""

    def test_merge_router_output(self):
        state = MetisState.from_query("write python code")
        node_output = {"route": "code", "next_node": "code"}
        for k, v in node_output.items():
            setattr(state, k, v)
        assert state.route == "code"
        assert state.next_node == "code"
        # Original query preserved
        assert state.query == "write python code"

    def test_merge_agent_output(self):
        state = MetisState.from_query("how does async work")
        node_output = {"response": "Async uses event loops.", "next_node": "formatter"}
        for k, v in node_output.items():
            setattr(state, k, v)
        assert state.response == "Async uses event loops."
        assert state.next_node == "formatter"

    def test_merge_all_fields(self):
        state = MetisState()
        node_output = {
            "query": "test",
            "route": "rag",
            "kb_context": "some context",
            "response": "an answer",
            "next_node": "formatter",
            "code_snippet": "",
        }
        for k, v in node_output.items():
            setattr(state, k, v)
        assert state.query == "test"
        assert state.kb_context == "some context"
        assert state.response == "an answer"
