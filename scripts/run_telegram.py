#!/usr/bin/env python3
"""Metis Telegram bot entry point — designed for systemd."""
import logging
import sys

# Configure root logger for systemd (stderr captured by journal)
root = logging.getLogger()
root.setLevel(logging.INFO)
if not root.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(h)

from dotenv import load_dotenv
load_dotenv(override=True)
from src.telegram.bot import run_polling
run_polling()
