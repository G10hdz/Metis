#!/usr/bin/env bash
# Metis Telegram bot starter for systemd
# Edit the path below to match your installation
exec /home/yourusername/Projects/Metis/.venv/bin/python -c \
"from dotenv import load_dotenv; load_dotenv(override=True); from src.telegram.bot import run_polling; run_polling()"
