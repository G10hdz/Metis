"""Obsidian vault writer — Karpathy pattern.

Writes structured notes with YAML frontmatter, wikilinks, and auto-updates index files.

Flow:
  1. LLM reads raw content (from search, articles, user queries)
  2. Writer creates structured markdown with frontmatter
  3. Note is saved to the appropriate wiki/ subfolder
  4. _Index.md files are updated with the new entry
"""

from __future__ import annotations

import logging
import re
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VAULT_ROOT = Path(os.getenv("METIS_VAULT_ROOT", Path.home() / ".metis" / "vault"))

# Domain → folder mapping
DOMAIN_FOLDERS: dict[str, str] = {
    "concept": "wiki/concepts",
    "tool": "wiki/tools",
    "pattern": "wiki/patterns",
    "project": "wiki/projects",
}

# Default folder if type classification fails
DEFAULT_FOLDER = "wiki/concepts"


def _slugify(text: str) -> str:
    """Convert title to filename-friendly slug."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def _ensure_dirs() -> None:
    """Ensure all vault directories exist."""
    for folder in DOMAIN_FOLDERS.values():
        (VAULT_ROOT / folder).mkdir(parents=True, exist_ok=True)
    (VAULT_ROOT / "raw").mkdir(parents=True, exist_ok=True)
    (VAULT_ROOT / "inbox").mkdir(parents=True, exist_ok=True)


def _update_index(folder_rel: str, title: str, summary: str, tags: list[str]) -> None:
    """
    Append an entry to the folder's _Index.md.
    Removes the "(empty)" placeholder if present.
    """
    index_path = VAULT_ROOT / folder_rel / "_Index.md"
    if not index_path.exists():
        # Create fresh index
        folder_name = folder_rel.split("/")[-1].title()
        index_path.write_text(
            f"# {folder_name} — Index\n\n"
            f"| Page | Summary | Tags |\n"
            f"|------|---------|------|\n"
            f"| [[{title}]] | {summary} | {', '.join(tags)} |\n",
            encoding="utf-8",
        )
        return

    content = index_path.read_text(encoding="utf-8")

    # Remove empty placeholder
    content = re.sub(r"\| _\(empty — no notes yet\)_ \|\| \|\n?", "", content)

    # Don't add duplicates
    if f"| [[{title}]] |" in content:
        logger.debug("Index entry already exists for '%s' in %s", title, folder_rel)
        return

    # Append before end
    content = content.rstrip() + f"\n| [[{title}]] | {summary} | {', '.join(tags)} |\n"
    index_path.write_text(content, encoding="utf-8")


def save_raw(query: str, content: str) -> Path:
    """Save unprocessed content to raw/ for later processing."""
    _ensure_dirs()
    ts = time.strftime("%Y%m%d-%H%M%S")
    slug = _slugify(query[:60])
    path = VAULT_ROOT / "raw" / f"{ts}-{slug}.md"
    path.write_text(
        f"# Raw: {query}\n\n"
        f"Source: Research query\n"
        f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n\n"
        f"{content}\n",
        encoding="utf-8",
    )
    logger.info("Raw note saved: %s", path)
    return path


def write_note(
    title: str,
    note_type: str,
    domain: str,
    tags: list[str],
    body: str,
    summary: str,
    related: list[str] | None = None,
    source: str = "",
) -> Path:
    """
    Write a structured Obsidian note with frontmatter and index update.

    Args:
        title: Note title (becomes [[wikilink]] target)
        note_type: "note" | "moc" | "pattern" | "tool" | "project"
        domain: "ai" | "devops" | "security" | "web" | "math"
        tags: List of tags (no # prefix — we add it)
        body: Markdown body (H2-sectioned content)
        summary: One-line summary for index tables
        related: List of existing note titles for "Related Pages" section
        source: URL or description of source material

    Returns:
        Path to the created note file.
    """
    _ensure_dirs()

    # Determine target folder
    folder_rel = DOMAIN_FOLDERS.get(note_type, DEFAULT_FOLDER)
    folder = VAULT_ROOT / folder_rel

    # Build slug
    slug = _slugify(title)
    ts = time.strftime("%Y%m%d")
    filename = f"{slug}.md"
    path = folder / filename

    # Build frontmatter
    tags_yaml = ", ".join(tags) if tags else ""
    fm = (
        f"---\n"
        f"title: \"{title}\"\n"
        f"type: {note_type}\n"
        f"domain: {domain}\n"
        f"tags: [{tags_yaml}]\n"
        f"created: {time.strftime('%Y-%m-%d')}\n"
        f"updated: {time.strftime('%Y-%m-%d')}\n"
    )
    if source:
        fm += f"source: \"{source}\"\n"
    fm += f"---\n\n"

    # Build body with related pages
    body_text = f"# {title}\n\n{body}"

    if related:
        body_text += "\n\n## Related Pages\n"
        for rel_title in related:
            body_text += f"- [[{rel_title}]]\n"

    full_content = fm + body_text + "\n"
    path.write_text(full_content, encoding="utf-8")

    # Update index
    display_tags = [f"#{t}" for t in (tags or [])]
    _update_index(folder_rel, title, summary, display_tags)

    logger.info("Note written: %s (%s/%s)", path, folder_rel, note_type)
    return path


def write_notes_batch(notes: list[dict[str, Any]]) -> list[Path]:
    """
    Write multiple notes and return paths.
    Each dict: {title, type, domain, tags, body, summary, related?, source?}
    """
    paths = []
    for note in notes:
        try:
            p = write_note(
                title=note["title"],
                note_type=note.get("type", "note"),
                domain=note.get("domain", "ai"),
                tags=note.get("tags", []),
                body=note["body"],
                summary=note.get("summary", note["title"]),
                related=note.get("related"),
                source=note.get("source", ""),
            )
            paths.append(p)
        except Exception as exc:
            logger.error("Failed to write note '%s': %s", note.get("title", "?"), exc)
    return paths
