"""Centralized configuration from .env and constants."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Telegram ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
ALLOWED_CHAT_IDS = os.getenv("ALLOWED_CHAT_IDS", "")
ALLOWED_CHAT_IDS_SET = (
    {int(x.strip()) for x in ALLOWED_CHAT_IDS.split(",") if x.strip()}
    if ALLOWED_CHAT_IDS
    else set()
)

# --- Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_BASE_URL_V1 = (
    OLLAMA_BASE_URL if OLLAMA_BASE_URL.endswith("/v1") else f"{OLLAMA_BASE_URL}/v1"
)

# Models
ROUTER_MODEL = os.getenv("METIS_ROUTER_MODEL", "phi3:mini")
CODE_MODEL = os.getenv("METIS_CODE_MODEL", "qwen2.5-coder:7b")
GENERAL_MODEL = os.getenv("METIS_GENERAL_MODEL", "phi3:mini")
EMBEDDING_MODEL = os.getenv("METIS_EMBEDDING_MODEL", "nomic-embed-text:latest")

# --- ChromaDB ---
CHROMA_DIR = Path(os.getenv("METIS_CHROMA_DIR", Path.home() / ".metis" / "chroma"))
CHROMA_COLLECTION = os.getenv("METIS_CHROMA_COLLECTION", "metis_kb")
RAG_SCORE_THRESHOLD = float(os.getenv("METIS_RAG_THRESHOLD", "0.65"))

# Optional second store (e.g., for a separate theory/academic vector store)
HYPATIA_CHROMA_DIR = os.getenv("METIS_HYPATIA_CHROMA_DIR", "").strip()
HYPATIA_CHROMA_COLLECTION = os.getenv("METIS_HYPATIA_CHROMA_COLLECTION", "hypatia_theory_chunks")

# --- VRAM ---
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "0s")  # configurable model warm time
VRAM_GUARD_LOG = Path(os.getenv("METIS_VRAM_LOG", Path.home() / ".metis" / "vram.log"))

# --- Fallback chain ---
FALLBACK_TIMEOUT = int(os.getenv("FALLBACK_TIMEOUT", "30"))
ERGANE_LOG = Path(os.getenv(
    "ERGANE_LOG",
    Path.home() / ".metis" / "ergane-logs.md"
))

# Opencode Zen API (HTTP, OpenAI-compatible)
OPENCODE_ZEN_API_KEY = os.getenv("OPENCODE_ZEN_API_KEY", "")
OPENCODE_ZEN_BASE_URL = "https://opencode.ai/zen/v1"
# Opencode models (short names — API doesn't use provider prefix)
OPENCODE_MODEL = os.getenv("METIS_OPENCODE_MODEL", "big-pickle")
OPENCODE_ALT_MODEL = os.getenv("METIS_OPENCODE_ALT_MODEL", "minimax-m2.5-free")

# Gemini API (free tier, quota-limited)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# File reader — allowed base directory
ALLOWED_FILE_ROOT = Path(os.getenv("METIS_FILE_ROOT", Path.home() / "Vscode-projects"))

# Allowed text file extensions for reading
ALLOWED_FILE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".mjs", ".cjs", ".json", ".jsonl",
    ".md", ".txt", ".rst", ".log",
    ".yaml", ".yml", ".toml", ".cfg", ".ini", ".env",
    ".sh", ".bash", ".zsh", ".fish",
    ".html", ".css", ".scss", ".sass", ".less",
    ".xml", ".csv", ".sql",
    ".gitignore", ".dockerignore", ".editorconfig",
    # Config / dotfiles
    ".conf", ".properties",
    # Binary formats with extractors
    ".pdf", ".docx", ".doc",
}

# Max file size to read (50MB — supports UNAM guides and large documents)
MAX_FILE_SIZE = int(os.getenv("METIS_MAX_FILE_SIZE", "52428800"))

# Bash command whitelist — only these commands are allowed
BASH_ALLOWED_COMMANDS = {
    "ls", "cat", "head", "tail", "wc", "find", "grep", "du", "df", "pwd",
    "tree", "stat", "diff", "sort", "uniq", "cut", "awk", "sed", "tr",
    "echo", "date", "whoami", "uname", "uptime", "free", "ps", "top",
    "md5sum", "sha256sum", "file", "which", "whereis", "basename", "dirname",
    "python", "python3",
}

# Dangerous patterns that are never allowed in bash commands
BASH_DANGEROUS = {"rm ", "sudo", "chmod ", "chown", "kill ", "dd ", "mkfs",
                   "wget ", "curl ", "apt ", "pip install", "> ", ">> ", "|",
                   "&&", "||", ";", "$(", "`", "eval", "exec", "source ",
                   "export ", "alias ", "unalias"}

# Max bash command output (20KB)
MAX_BASH_OUTPUT = int(os.getenv("METIS_MAX_BASH_OUTPUT", "20480"))

# Bash timeout (seconds)
BASH_TIMEOUT = int(os.getenv("METIS_BASH_TIMEOUT", "10"))

# Ensure dirs exist
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
VRAM_GUARD_LOG.parent.mkdir(parents=True, exist_ok=True)
ERGANE_LOG.parent.mkdir(parents=True, exist_ok=True)
