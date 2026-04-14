"""LangGraph-compatible StateGraph state definition using Pydantic v2."""

from __future__ import annotations

from typing import Annotated, Optional
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class MetisState(BaseModel):
    """
    Shared state across all graph nodes.

    Every node MUST return a dict whose keys are a subset of these fields.
    LangGraph merges returned dicts into the accumulated state automatically.
    """

    query: str = Field(default="", description="Original user query")
    messages: Annotated[list, add_messages] = Field(
        default_factory=list, description="Conversation message history"
    )
    next_node: str = Field(
        default="router", description="Next node to route to (used by conditional edges)"
    )
    kb_context: str = Field(default="", description="Retrieved knowledge-base context from RAG")
    response: str = Field(default="", description="Final formatted response text")
    route: str = Field(default="", description="Classification label set by router")
    code_snippet: str = Field(default="", description="Code produced by code_agent")
    search_context: str = Field(
        default="",
        description="JSON-serialized search state for resumable deep research (learnings, sources, queries)",
    )
    source: str = Field(
        default="cli",
        description="Interaction source: 'telegram' | 'web' | 'cli'",
    )
    file_path: str = Field(default="", description="Resolved file path requested by user")
    file_content: str = Field(default="", description="Contents of the read file")
    file_error: str = Field(default="", description="Error message if file read fails")
    bash_output: str = Field(default="", description="Output from bash command execution")
    bash_error: str = Field(default="", description="Error from bash command execution")
    awaiting_confirmation: bool = Field(default=False, description="Whether we're waiting for user confirmation")
    pending_action: str = Field(default="", description="Serialized pending action (for delete/edit confirmation)")

    # --- helpers ---
    def to_dict(self) -> dict:
        """Serialize to a plain dict (LangGraph merge-compatible)."""
        return self.model_dump(exclude_unset=True)

    @classmethod
    def from_query(cls, query: str) -> "MetisState":
        """Create an initial state from a raw query string."""
        return cls(query=query, messages=[{"role": "user", "content": query}])
