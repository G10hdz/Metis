"""PDF/DOCX ingestion → study questions + ChromaDB chunks.

Uses Marker (VikParuchuri) for PDF → Markdown with LaTeX equation support
on math-heavy PDFs. Fallback to pypdf/pdfplumber for simple text.

Usage:
    python -m src.memory.ingest /path/to/pdfs/
    python -m src.memory.ingest ~/Documents/study-materials/
    python -m src.memory.ingest ~/Documents/study-materials/ --marker

Output:
    - questions.json (study questions extracted from material)
    - ChromaDB populated with study material chunks
    - ingestion_report.txt with stats
"""

from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---- Data structures ----

@dataclass
class Question:
    id: str
    materia: str
    tema: str
    dificultad: int
    pregunta: str
    opciones: list[str]
    correcta: int
    explicacion: str = ""
    tip: str = ""

    def to_js_object(self) -> str:
        opts = json.dumps(self.opciones, ensure_ascii=False)
        return (
            f'  {{\n'
            f'    id: "{self.id}",\n'
            f'    materia: "{self.materia}",\n'
            f'    tema: "{self.tema}",\n'
            f'    dificultad: {self.dificultad},\n'
            f'    pregunta: {json.dumps(self.pregunta, ensure_ascii=False)},\n'
            f'    opciones: {opts},\n'
            f'    correcta: {self.correcta},\n'
            f'    explicacion: {json.dumps(self.explicacion, ensure_ascii=False)},\n'
            f'    tip: {json.dumps(self.tip, ensure_ascii=False)}\n'
            f'  }}'
        )


@dataclass
class IngestionResult:
    questions: list[Question] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)
    files_processed: int = 0
    files_failed: int = 0


# ---- Text extractors ----

_marker_model_cache = None
_MARKER_AVAILABLE = False


def _check_marker_available() -> bool:
    global _MARKER_AVAILABLE
    if _MARKER_AVAILABLE:
        return True
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        _MARKER_AVAILABLE = True
        return True
    except Exception:
        return False


def _get_marker_converter():
    global _marker_model_cache
    if _marker_model_cache is None:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        _marker_model_cache = PdfConverter(artifact_dict=create_model_dict())
        logger.info("Marker models loaded")
    return _marker_model_cache


def _marker_extract(path: Path) -> str:
    if not _check_marker_available():
        return ""
    try:
        converter = _get_marker_converter()
        rendered = converter(str(path))
        from marker.output import text_from_rendered
        text, _, _ = text_from_rendered(rendered)
        return text or ""
    except Exception as exc:
        logger.warning("Marker failed on %s: %s", path.name, exc)
        return ""


def _is_math_heavy(path: Path) -> bool:
    """Quick heuristic for math-heavy PDFs."""
    name = path.name.lower()
    if any(kw in name for kw in ["matematica", "mate", "fisica", "algebra", "trigonometria", "geometria"]):
        return True
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        sample = ""
        for page in reader.pages[:2]:
            sample += (page.extract_text() or "")
        symbol_count = sum(1 for c in sample if c in "²³√∑∫∂Δπθλ±≤≥≠")
        line_count = max(sample.count("\n"), 1)
        return symbol_count / line_count > 0.3
    except Exception:
        return False


def extract_pdf_text(path: Path, use_marker: bool = False) -> str:
    """
    Extract text from PDF.
    - use_marker=True: Marker (slow, LaTeX equations)
    - use_marker=False: pypdf → pdfplumber (fast)
    - Auto-detects math-heavy PDFs and tries Marker if available
    """
    if use_marker or _is_math_heavy(path):
        text = _marker_extract(path)
        if text.strip():
            return text

    text = ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        for page in reader.pages:
            pt = page.extract_text() or ""
            if pt.strip():
                text += pt + "\n\n"
    except Exception as exc:
        logger.warning("pypdf failed on %s: %s", path.name, exc)

    if not text.strip():
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    pt = page.extract_text() or ""
                    text += pt + "\n\n"
        except Exception as exc:
            logger.warning("pdfplumber also failed on %s: %s", path.name, exc)

    return text


def extract_docx_text(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as exc:
        logger.warning("DOCX extraction failed on %s: %s", path.name, exc)
        return ""


# ---- Question parsers ----

def parse_unam_questions(text: str) -> list[dict[str, Any]]:
    """
    Parse UNAM exam-style questions. Handles:
    - "1. ¿Pregunta?\n\nA) Opción\nB) Opción\nC) Opción\nD) Opción"
    - Answer key sections ("RESPUESTAS: 1.a 2.b ...")
    """
    questions: list[dict[str, Any]] = []

    # Find all option lines: "A) text" or "a) text"
    opt_line = re.compile(r'^\s*([a-dA-D])\)\s*(.+)$', re.MULTILINE)
    option_positions: list[tuple[int, str, str]] = []
    for m in opt_line.finditer(text):
        option_positions.append((m.start(), m.group(1), m.group(2).strip()))

    # Group consecutive options into blocks (within 100 chars of each other)
    option_blocks: list[tuple[int, list[tuple[str, str]]]] = []
    current: list[tuple[str, str]] = []
    block_start = -1
    for pos, letter, opt_text in option_positions:
        if not current or (pos - block_start < 300 and len(current) < 4):
            if not current:
                block_start = pos
            current.append((letter, opt_text))
        else:
            if len(current) >= 2:
                option_blocks.append((block_start, current))
            block_start = pos
            current = [(letter, opt_text)]
    if len(current) >= 2:
        option_blocks.append((block_start, current))

    # Find numbered questions and match with nearest option block after them
    q_start = re.compile(r'(?:^|\n)\s*(\d+)[\.\)]\s*(.+)', re.MULTILINE)
    for q_match in q_start.finditer(text):
        q_num = q_match.group(1)
        q_text = q_match.group(2).strip()
        q_pos = q_match.start()

        # Find closest option block AFTER the question (within 500 chars)
        best_block = None
        best_dist = float('inf')
        for block_pos, block_opts in option_blocks:
            dist = block_pos - q_pos
            if 0 < dist < 500 and dist < best_dist:
                best_block = block_opts
                best_dist = dist

        if best_block:
            opciones = [t for _, t in best_block[:4]]
            questions.append({
                "raw_num": q_num,
                "pregunta": q_text,
                "opciones": opciones,
                "correcta": -1,
            })

    # Answer key extraction
    answer_key = _extract_answer_key(text)
    for q in questions:
        if q["raw_num"] in answer_key:
            q["correcta"] = answer_key[q["raw_num"]]

    return questions


def _extract_answer_key(text: str) -> dict[str, int]:
    """Extract answer key from text sections like 'RESPUESTAS: 1.a 2.b'."""
    answers: dict[str, int] = {}
    letter_to_idx = {"a": 0, "b": 1, "c": 2, "d": 3, "A": 0, "B": 1, "C": 2, "D": 3}

    key_section = re.search(
        r'(?:RESPUESTAS|RESPUESTA|Clave|clave|ANSWERS|Hoja de respuestas)[\s:\-]*\n(.*?)(?=\n\n[A-Z]|\Z)',
        text, re.DOTALL
    )
    if key_section:
        section_text = key_section.group(1)
        for m in re.finditer(r'(\d+)\s*[.\)]?\s*([a-dA-D])', section_text):
            num, letter = m.group(1), m.group(2)
            if letter in letter_to_idx:
                answers[num] = letter_to_idx[letter]

    return answers


def detect_materia(filename: str) -> str:
    """Detect subject from filename."""
    name = filename.lower()
    mapping = {
        "matematica": "matematicas", "mate": "matematicas",
        "aritm": "matematicas", "algebra": "matematicas",
        "geometr": "matematicas", "trigonometria": "matematicas",
        "fisica": "fisica", "quimica": "quimica", "biologia": "biologia",
        "espanol": "espanol", "español": "espanol",
        "literatura": "literatura",
        "historia de mexico": "historia_mx", "historia de méxico": "historia_mx",
        "historia mexico": "historia_mx", "historia méxico": "historia_mx",
        "historia universal": "historia_uni", "historia uni": "historia_uni",
        "geografia": "geografia", "geografía": "geografia",
        "filosofia": "filosofia", "filosofía": "filosofia",
    }
    for key, value in mapping.items():
        if key in name:
            return value
    return "general"


def detect_tema(text: str, materia: str) -> str:
    """Detect topic from question text."""
    text_lower = text.lower()
    topics = {
        "matematicas": {
            "algebra": ["x²", "ecuación", "factoriz", "polinom", "raíz"],
            "aritmetica": ["porcentaje", "fracción", "MCM", "divisible", "número"],
            "geometria_analitica": ["pendiente", "circunferencia", "parábola", "distancia"],
            "trigonometria": ["seno", "coseno", "tangente", "triángulo", "ángulo", "sen("],
            "probabilidad": ["probabilidad", "dado", "moneda", "azar"],
        },
        "fisica": {
            "cinematica": ["velocidad", "aceleración", "distancia", "caída", "tiempo"],
            "leyes_newton": ["fuerza", "newton", "masa", "aceleración", "normal"],
            "energia": ["energía", "trabajo", "joule", "cinética", "potencial"],
            "electricidad": ["voltaje", "corriente", "resistencia", "circuito"],
        },
    }
    subject_topics = {}
    if materia == "matematicas":
        subject_topics = topics.get("matematicas", {})
    elif materia == "fisica":
        subject_topics = topics.get("fisica", {})
    else:
        return "general"

    for tema, keywords in subject_topics.items():
        if any(kw in text_lower for kw in keywords):
            return tema
    return "general"


# ---- Question generation from theory (via Metis) ----

def generate_questions_from_text(text: str, materia: str, num_questions: int = 5) -> list[Question]:
    """Generate MCQ questions from theory text using the Metis graph."""
    from src.graph.orchestrator import get_graph
    from src.graph.state import MetisState

    truncated = text[:3000]
    prompt = (
        f"Basándote en el siguiente texto, genera {num_questions} preguntas de opción múltiple "
        f"estilo examen de admisión UNAM para {materia}.\n\n"
        f"Texto: {truncated}\n\n"
        f"Formato: JSON array con pregunta, opciones (4), correcta (0-3), explicacion, tema."
    )

    try:
        graph = get_graph()
        state = MetisState.from_query(prompt)
        result = graph.invoke(state.model_dump())
        response = result.get("response", "")

        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            raw = json.loads(json_match.group(0))
            questions = []
            for i, q in enumerate(raw):
                questions.append(Question(
                    id=f"{materia[:3]}_gen_{i+1:03d}",
                    materia=materia,
                    tema=q.get("tema", "general"),
                    dificultad=2,
                    pregunta=q.get("pregunta", ""),
                    opciones=q.get("opciones", []),
                    correcta=q.get("correcta", 0),
                    explicacion=q.get("explicacion", ""),
                    tip="",
                ))
            return questions
    except Exception as exc:
        logger.warning("Failed to generate questions from text: %s", exc)

    return []


# ---- Chunking for ChromaDB ----

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks for vector DB."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end - overlap
    return chunks


# ---- Export to questions.js ----

def export_questions_js(questions: list[Question], output_path: Path, existing_ids: set[str] | None = None) -> None:
    """Write questions.json file with study questions extracted from material."""
    existing_ids = existing_ids or set()
    seen_ids: set[str] = set()
    unique: list[Question] = []
    for q in questions:
        if q.id not in seen_ids and q.id not in existing_ids:
            seen_ids.add(q.id)
            unique.append(q)

    import time
    js_objects = "\n,\n".join(q.to_js_object() for q in unique)
    js_content = (
        f"// questions.js — Banco de preguntas PumaPrep\n"
        f"// Generado por Metis v2.3 el {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"// {len(unique)} preguntas\n\n"
        f"window.QUESTIONS = [\n{js_objects}\n];\n\n"
        f"window.PREGUNTAS_POR_MATERIA = (materia) =>\n"
        f"  window.QUESTIONS.filter(q => q.materia === materia);\n\n"
        f"window.PREGUNTAS_POR_TEMA = (materia, tema) =>\n"
        f"  window.QUESTIONS.filter(q => q.materia === materia && q.tema === tema);\n\n"
        f"window.PREGUNTAS_ALEATORIAS = (n, materia = null) => {{\n"
        f"  let pool = materia ? window.PREGUNTAS_POR_MATERIA(materia) : window.QUESTIONS;\n"
        f"  return pool.sort(() => Math.random() - 0.5).slice(0, n);\n"
        f"}};\n\n"
        f"console.log(`[PumaPrep] Banco cargado: ${{window.QUESTIONS.length}} preguntas`);\n"
    )
    output_path.write_text(js_content, encoding="utf-8")
    logger.info("Wrote %d questions to %s", len(unique), output_path)


# ---- Main ingestion pipeline ----

def ingest_directory(directory: Path, output_dir: Path | None = None, use_marker: bool = False) -> IngestionResult:
    """
    Scan directory for PDFs/DOCX, extract questions and chunks.

    Args:
        directory: Path to scan for PDF/DOCX files
        output_dir: Where to write questions.js (defaults to directory)
        use_marker: Use Marker for all PDFs (slow but preserves LaTeX)
    """
    if output_dir is None:
        output_dir = directory

    result = IngestionResult()
    all_questions: list[Question] = []
    all_chunks: list[str] = []

    # Load existing questions
    existing_js = output_dir / "questions.js"
    existing_ids: set[str] = set()
    if existing_js.exists():
        content = existing_js.read_text(encoding="utf-8")
        for m in re.finditer(r'id:\s*"([^"]+)"', content):
            existing_ids.add(m.group(1))

    # Find files
    files = sorted(
        list(directory.glob("*.pdf")) +
        list(directory.glob("*.docx")) +
        list(directory.glob("*.DOCX"))
    )
    logger.info("Found %d files to process in %s", len(files), directory)

    for filepath in files:
        logger.info("Processing %s...", filepath.name)

        if filepath.suffix.lower() == ".pdf":
            text = extract_pdf_text(filepath, use_marker=use_marker)
        elif filepath.suffix.lower() == ".docx":
            text = extract_docx_text(filepath)
        else:
            continue

        if not text.strip():
            logger.warning("  No text extracted from %s", filepath.name)
            result.files_failed += 1
            continue

        result.files_processed += 1

        # Try to parse existing questions
        parsed = parse_unam_questions(text)
        if parsed:
            logger.info("  Found %d questions in %s", len(parsed), filepath.name)
            materia = detect_materia(filepath.name)
            for p in parsed:
                opciones = p.get("opciones", [])
                if len(opciones) < 2:
                    continue
                tema = detect_tema(p.get("pregunta", ""), materia)
                correcta = p.get("correcta", 0)
                if correcta < 0:
                    correcta = 0
                q = Question(
                    id=f"{materia[:3]}_{filepath.stem.replace(' ', '_')[:10]}_{p['raw_num']}",
                    materia=materia,
                    tema=tema,
                    dificultad=2,
                    pregunta=p["pregunta"],
                    opciones=opciones,
                    correcta=correcta,
                )
                all_questions.append(q)
        else:
            # No structured questions — chunk for RAG
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            logger.info("  Chunked into %d pieces for RAG", len(chunks))

            # Generate questions from theory (limit 2 per file)
            if len(text) > 200:
                materia = detect_materia(filepath.name)
                generated = generate_questions_from_text(text, materia, num_questions=2)
                if generated:
                    logger.info("  Generated %d questions from theory", len(generated))
                    all_questions.extend(generated)

    # Add to ChromaDB
    if all_chunks:
        from src.memory.store import get_store
        store = get_store()
        ids = [f"chunk_{uuid.uuid4().hex[:8]}" for _ in all_chunks]
        metadatas = [{"source": "ingestion"} for _ in all_chunks]
        store.add(documents=all_chunks, ids=ids, metadatas=metadatas)
        logger.info("Added %d chunks to ChromaDB", len(all_chunks))

    result.questions = all_questions
    result.chunks = all_chunks

    # Export questions.js
    export_questions_js(all_questions, output_dir / "questions.js", existing_ids)

    # Write report
    import time
    materias = {}
    for q in all_questions:
        materias[q.materia] = materias.get(q.materia, 0) + 1

    report_path = output_dir / "ingestion_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Ingestion Report — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Files processed: {result.files_processed}\n")
        f.write(f"Files failed: {result.files_failed}\n")
        f.write(f"Questions extracted/generated: {len(all_questions)}\n")
        f.write(f"Chunks added to ChromaDB: {len(all_chunks)}\n\n")
        f.write("Questions by subject:\n")
        for m, c in sorted(materias.items()):
            f.write(f"  {m}: {c}\n")

    logger.info("Ingestion complete! Report: %s", report_path)
    return result


# ---- CLI entry point ----

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest PDFs into Metis ChromaDB + generate study questions")
    parser.add_argument("directory", help="Path to directory containing PDFs/DOCX files")
    parser.add_argument("--marker", action="store_true", help="Use Marker for all PDFs (slow, preserves LaTeX)")
    args = parser.parse_args()

    target = Path(args.directory)
    if not target.exists():
        print(f"Error: {target} does not exist")
        sys.exit(1)

    ingest_directory(target, use_marker=args.marker)
