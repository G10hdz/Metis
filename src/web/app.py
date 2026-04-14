"""Gradio web interface — Chat + Dashboard for Metis v2."""

from __future__ import annotations

import json
import logging
import time

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.graph.orchestrator import get_graph
from src.graph.state import MetisState
from src.telemetry.store import get_telemetry

logger = logging.getLogger(__name__)

# ---------- Chat tab ----------

def _run_graph(query: str, history: list[dict], chat_source: str = "web") -> tuple[str, list[dict]]:
    """Run the Metis graph and return (response_text, updated_history)."""
    try:
        graph = get_graph()
        initial = MetisState.from_query(query)
        initial.source = chat_source
        result: dict = graph.invoke(initial.model_dump())
        response = result.get("response", "No response generated.")
        tier = result.get("_tier", "ollama")  # fallback tier label

        # Add tier badge
        tier_badge = f"📡 [{tier}]" if tier and tier != "ollama" else ""
        if tier_badge:
            response = f"{tier_badge}\n\n{response}"

        history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": response}]
        return response, history
    except Exception as exc:
        logger.exception("Graph execution failed: %s", exc)
        error_msg = f"⚠️ Error: {exc}"
        history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": error_msg}]
        return error_msg, history


def _deeper_search(history: list[dict], last_response: str) -> tuple[str, list[dict]]:
    """Send 'deeper' to continue the last search."""
    return _run_graph("go deeper", history)


# ---------- Dashboard tab ----------

def _render_dashboard() -> tuple:
    """Return all dashboard components."""
    telem = get_telemetry()
    stats = telem.stats()
    recent = telem.recent(20)
    latency_data = telem.latency_series(100)

    # Stats cards
    total = stats["total"]
    errors = stats["errors"]
    avg_lat = stats["avg_latency_ms"]

    # Route distribution table
    routes = stats.get("routes", {})
    route_rows = [[k, v] for k, v in routes.items()]

    # Recent conversations table
    recent_rows = [
        [r["ts"][:19], r["route"], r["source"], r["query"][:60], f'{r["latency_ms"]:.0f}ms', r["error"][:30] if r["error"] else ""]
        for r in recent
    ]

    # Latency chart
    fig, ax = plt.subplots(figsize=(10, 4))
    if latency_data:
        lats = [r["latency_ms"] for r in reversed(latency_data)]
        labels = list(range(len(lats)))
        ax.bar(labels, lats, color="#4A90D9", alpha=0.8)
        ax.set_xlabel("Conversation #")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Query Latency")
        ax.set_xticks([])
    else:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)

    return (
        f"### 📊 Stats\n- **Total:** {total}\n- **Errors:** {errors}\n- **Avg Latency:** {avg_lat}ms",
        route_rows,
        recent_rows,
        fig,
    )


def _build_ui() -> gr.Blocks:
    """Build the full Gradio UI with Chat + Dashboard tabs."""
    with gr.Blocks(title="Metis v2", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🧠 Metis v2 — AI Agent Orchestrator")

        with gr.Tabs():
            # ---- Chat Tab ----
            with gr.Tab("💬 Chat"):
                chatbot = gr.Chatbot(height=450)
                with gr.Row():
                    msg = gr.Textbox(placeholder="Ask Metis anything...", scale=4, show_label=False)
                    send_btn = gr.Button("Send", scale=1)
                deeper_btn = gr.Button("🔍 Go Deeper", variant="secondary")

                chat_state = gr.State(value=[])

                def on_submit(message: str, history: list[dict]):
                    response, new_history = _run_graph(message, history, chat_source="web")
                    return "", new_history

                msg.submit(on_submit, [msg, chat_state], [msg, chat_state])
                send_btn.click(on_submit, [msg, chat_state], [msg, chat_state])
                deeper_btn.click(_deeper_search, [chat_state], [chat_state])

            # ---- Dashboard Tab ----
            with gr.Tab("📊 Dashboard"):
                stats_md = gr.Markdown()
                with gr.Row():
                    with gr.Column():
                        route_df = gr.Dataframe(headers=["Route", "Count"], label="Route Distribution")
                    with gr.Column():
                        latency_plot = gr.Plot(label="Latency Over Time")
                recent_df = gr.Dataframe(
                    headers=["Time", "Route", "Source", "Query", "Latency", "Error"],
                    label="Recent Conversations",
                    wrap=True,
                )
                refresh_btn = gr.Button("🔄 Refresh")
                refresh_btn.click(_render_dashboard, outputs=[stats_md, route_df, recent_df, latency_plot])

        # Initial load
        app.load(_render_dashboard, outputs=[stats_md, route_df, recent_df, latency_plot])

    return app


def run_web(share: bool = False, port: int = 7860) -> None:
    """Launch the Gradio web interface."""
    app = _build_ui()
    logger.info("Starting Gradio web UI on port %d", port)
    app.launch(server_name="0.0.0.0", server_port=port, share=share)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_web()
