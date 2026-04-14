"""Metis v2 — single entry point. Starts Telegram + Gradio simultaneously."""

from __future__ import annotations

import logging
import sys
import threading

from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_all() -> None:
    """Start both Telegram bot and Gradio web UI in the same process."""
    logger.info("=" * 60)
    logger.info("Metis v2 — AI Agent Orchestrator")
    logger.info("=" * 60)

    # Pre-init the graph (loads models on first call)
    logger.info("Pre-warming graph...")
    from src.graph.orchestrator import get_graph
    get_graph()
    logger.info("Graph ready.")

    # Pre-init telemetry
    from src.telemetry.store import get_telemetry
    get_telemetry()
    logger.info("Telemetry ready.")

    # Start Gradio in a background thread
    def start_gradio():
        from src.web.app import run_web
        try:
            run_web(port=7860)
        except Exception as exc:
            logger.error("Gradio failed to start: %s", exc)

    gradio_thread = threading.Thread(target=start_gradio, daemon=True)
    gradio_thread.start()
    logger.info("Gradio web UI → http://localhost:7860")

    # Start Telegram (blocking) in main thread
    if settings.TELEGRAM_TOKEN:
        logger.info("Starting Telegram bot...")
        from src.telegram.bot import run_polling
        run_polling()
    else:
        logger.warning("TELEGRAM_TOKEN not set — running web UI only.")
        logger.info("Set TELEGRAM_TOKEN in .env to enable Telegram bot.")
        # Keep main thread alive so daemon thread runs
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down.")


if __name__ == "__main__":
    run_all()
