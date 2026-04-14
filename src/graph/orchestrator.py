"""LangGraph StateGraph orchestration — builds and compiles the Metis graph."""

from __future__ import annotations

import logging
from langgraph.graph import StateGraph, END

from src.graph.state import MetisState
from src.graph.nodes import (
    ROUTE_RAG,
    ROUTE_CODE,
    ROUTE_SEARCH,
    ROUTE_SEARCH_CONTINUE,
    ROUTE_FILE,
    ROUTE_FILE_EDIT,
    ROUTE_FILE_DELETE,
    ROUTE_BASH,
    ROUTE_GENERAL,
    ROUTE_RESEARCH,
    ROUTE_PRACTICE,
    router,
    rag_agent,
    code_agent,
    search_agent,
    general_agent,
    research_agent,
    file_reader_agent,
    file_editor_agent,
    file_deleter_agent,
    bash_agent,
    echo_agent,
    formatter,
    route_decision,
)

logger = logging.getLogger(__name__)


def build_graph() -> "CompiledGraph":
    """
    Construct and compile the Metis StateGraph.

    Graph flow:
        __start__ → router → [rag | code | search | search_continue | research | general] → formatter → __end__
    """
    graph = StateGraph(MetisState)

    # Add nodes
    graph.add_node("router", router)
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("code_agent", code_agent)
    graph.add_node("search_agent", search_agent)
    graph.add_node("search_resume", lambda state: search_agent(state, resume=True))
    graph.add_node("general_agent", general_agent)
    graph.add_node("research_agent", research_agent)
    graph.add_node("file_reader_agent", file_reader_agent)
    graph.add_node("file_editor_agent", file_editor_agent)
    graph.add_node("file_deleter_agent", file_deleter_agent)
    graph.add_node("bash_agent", bash_agent)
    graph.add_node("echo_agent", echo_agent)
    graph.add_node("formatter", formatter)

    # Entry point
    graph.set_entry_point("router")

    # Conditional edges from router
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            ROUTE_RAG: "rag_agent",
            ROUTE_CODE: "code_agent",
            ROUTE_SEARCH: "search_agent",
            ROUTE_SEARCH_CONTINUE: "search_resume",
            ROUTE_FILE: "file_reader_agent",
            ROUTE_FILE_EDIT: "file_editor_agent",
            ROUTE_FILE_DELETE: "file_deleter_agent",
            ROUTE_BASH: "bash_agent",
            ROUTE_RESEARCH: "research_agent",
            ROUTE_PRACTICE: "echo_agent",
            ROUTE_GENERAL: "general_agent",
            "formatter": "formatter",  # edge case: empty query
        },
    )

    # All agents flow to formatter
    graph.add_edge("rag_agent", "formatter")
    graph.add_edge("code_agent", "formatter")
    graph.add_edge("search_agent", "formatter")
    graph.add_edge("search_resume", "formatter")
    graph.add_edge("general_agent", "formatter")
    graph.add_edge("research_agent", "formatter")
    graph.add_edge("file_reader_agent", "formatter")
    graph.add_edge("file_editor_agent", "formatter")
    graph.add_edge("file_deleter_agent", "formatter")
    graph.add_edge("bash_agent", "formatter")
    graph.add_edge("echo_agent", "formatter")

    # Formatter ends the graph
    graph.add_edge("formatter", END)

    compiled = graph.compile()
    logger.info("Metis graph compiled successfully")
    return compiled


# Module-level singleton — lazy init on first access
_compiled_graph = None


def get_graph():
    """Return the compiled graph (lazy singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# Type alias for the return type of build_graph
try:
    from langgraph.graph.graph import CompiledGraph  # type: ignore
except ImportError:
    CompiledGraph = object  # type: ignore
