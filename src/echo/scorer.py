"""Echo scoring engine — word-level Levenshtein alignment with confidence scoring."""

from __future__ import annotations

import logging
from typing import Any

import Levenshtein

logger = logging.getLogger(__name__)


class EchoScorer:
    """
    Scores user pronunciation by comparing expected vs actual transcription.
    
    Uses full-string Levenshtein ratio for overall accuracy,
    then word-level alignment for per-word feedback.
    """

    def __init__(
        self,
        correct_threshold: float = 0.85,
        partial_threshold: float = 0.6,
    ):
        self.correct_threshold = correct_threshold
        self.partial_threshold = partial_threshold

    def score(self, expected: str, actual: str) -> dict[str, Any]:
        """
        Compare expected vs actual transcription.
        
        Returns:
            {
                "overall_score": 0-100,
                "grade": "A" | "B" | "C" | "D",
                "words": [
                    {"word": "the", "status": "correct" | "partial" | "incorrect" | "missed", "said": "the"},
                    ...
                ],
                "flagged": [{"word": "comfortable", "status": "incorrect", "said": "comfortble"}]
            }
        """
        if not expected.strip():
            return {"overall_score": 0, "grade": "F", "words": [], "flagged": []}

        # Overall accuracy (full-string Levenshtein)
        overall_accuracy = Levenshtein.ratio(expected.lower().strip(), actual.lower().strip())
        overall_score = round(overall_accuracy * 100)

        # Word-level alignment
        expected_words = expected.lower().strip().split()
        actual_words = actual.lower().strip().split()

        word_results = []
        flagged = []
        
        actual_idx = 0
        
        for exp_word in expected_words:
            if actual_idx >= len(actual_words):
                # User stopped speaking — mark remaining as missed
                word_results.append({"word": exp_word, "status": "missed", "said": ""})
                flagged.append({"word": exp_word, "status": "missed", "said": "(silence)"})
                continue

            # Find best match for this expected word in actual words
            best_ratio = 0.0
            best_actual = actual_words[actual_idx]
            best_idx = actual_idx

            # Look at current and next 2 words (handles insertions/deletions)
            for offset in range(3):
                if actual_idx + offset < len(actual_words):
                    candidate = actual_words[actual_idx + offset]
                    ratio = Levenshtein.ratio(exp_word, candidate)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_actual = candidate
                        best_idx = actual_idx + offset

            # Determine status
            if best_ratio >= self.correct_threshold:
                status = "correct"
            elif best_ratio >= self.partial_threshold:
                status = "partial"
            else:
                status = "incorrect"

            word_results.append({"word": exp_word, "status": status, "said": best_actual})

            if status in ("incorrect", "missed"):
                flagged.append({"word": exp_word, "status": status, "said": best_actual})

            # Advance actual index
            actual_idx = best_idx + 1

        # Calculate grade
        correct_count = sum(1 for w in word_results if w["status"] == "correct")
        total = len(word_results)
        accuracy_ratio = correct_count / total if total > 0 else 0

        if accuracy_ratio >= 0.9:
            grade = "A"
        elif accuracy_ratio >= 0.7:
            grade = "B"
        elif accuracy_ratio >= 0.5:
            grade = "C"
        else:
            grade = "D"

        return {
            "overall_score": overall_score,
            "grade": grade,
            "words": word_results,
            "flagged": flagged,
        }

    def format_feedback(self, score_result: dict[str, Any]) -> str:
        """Format scoring result into user-friendly feedback."""
        score = score_result["overall_score"]
        grade = score_result["grade"]
        words = score_result["words"]
        flagged = score_result["flagged"]

        # Emoji indicators
        if score >= 90:
            emoji = "🌟"
        elif score >= 70:
            emoji = "👍"
        elif score >= 50:
            emoji = "💪"
        else:
            emoji = "🔄"

        feedback = f"{emoji} **Score: {score}% (Grade: {grade})**\n\n"

        # Word-by-word breakdown
        word_display = []
        for w in words:
            if w["status"] == "correct":
                word_display.append(f"🟢 {w['word']}")
            elif w["status"] == "partial":
                word_display.append(f"🟠 {w['word']}")
            elif w["status"] == "missed":
                word_display.append(f"⚪ {w['word']} (missed)")
            else:
                word_display.append(f"🔴 {w['word']} (said: {w['said']})")

        feedback += " ".join(word_display)
        feedback += "\n\n"

        # Flagged words summary
        if flagged:
            feedback += "**Words to practice:**\n"
            for f in flagged:
                if f["status"] == "missed":
                    feedback += f"• {f['word']} — {f['said']}\n"
                else:
                    feedback += f"• {f['word']} → you said: {f['said']}\n"
        else:
            feedback += "✨ Perfect! All words pronounced correctly!"

        return feedback
