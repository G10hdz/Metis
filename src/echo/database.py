"""Echo database — SQLite persistence for practice sessions and progress."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import sqlite3

logger = logging.getLogger(__name__)

DB_PATH = os.path.expanduser("~/.metis/echo.db")


class EchoDatabase:
    """
    SQLite database for Echo practice sessions and user progress.
    
    Tables:
    - practice_sessions: Individual practice attempts
    - user_progress: Aggregated user stats
    - sentence_library: Available practice sentences
    - user_word_stats: Per-word performance tracking
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_connection()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS practice_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    target_sentence TEXT NOT NULL,
                    actual_transcription TEXT,
                    score INTEGER,
                    grade TEXT,
                    flagged_words TEXT,
                    language TEXT DEFAULT 'en',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS user_progress (
                    user_id TEXT PRIMARY KEY,
                    level TEXT DEFAULT 'A1',
                    total_sessions INTEGER DEFAULT 0,
                    avg_score REAL DEFAULT 0.0,
                    streak_days INTEGER DEFAULT 0,
                    last_practice DATE,
                    total_words_practiced INTEGER DEFAULT 0,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS sentence_library (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL UNIQUE,
                    language TEXT NOT NULL DEFAULT 'en',
                    level TEXT DEFAULT 'A1',
                    topic TEXT,
                    times_practiced INTEGER DEFAULT 0,
                    avg_difficulty REAL DEFAULT 0.5
                );

                CREATE TABLE IF NOT EXISTS user_word_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    word TEXT NOT NULL,
                    language TEXT DEFAULT 'en',
                    times_attempted INTEGER DEFAULT 0,
                    times_correct INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    last_attempted DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, word, language)
                );

                -- Seed some practice sentences
                INSERT OR IGNORE INTO sentence_library (text, language, level, topic) VALUES
                    ('The comfortable chair was near the door', 'en', 'A1', 'daily'),
                    ('She sells seashells by the seashore', 'en', 'A2', 'tongue-twister'),
                    ('The weather is beautiful today', 'en', 'A1', 'daily'),
                    ('I would like to order a cup of coffee', 'en', 'A1', 'restaurant'),
                    ('The photograph shows a magnificent landscape', 'en', 'B1', 'descriptive'),
                    ('He thoroughly enjoyed the thrilling adventure', 'en', 'B2', 'narrative'),
                    ('La biblioteca está cerrada los domingos', 'es', 'A1', 'daily'),
                    ('Me gustaría reservar una mesa para dos', 'es', 'A2', 'restaurant'),
                    ('El clima está muy agradable hoy', 'es', 'A1', 'daily'),
                    ('Los niños juegan en el parque central', 'es', 'A2', 'daily');
            """)
            conn.commit()
            logger.info(f"Echo database initialized at {self.db_path}")
        except Exception as exc:
            logger.error(f"Failed to initialize database: {exc}")
            conn.rollback()
        finally:
            conn.close()

    def save_session(self, user_id: str, target_sentence: str, score: int,
                     grade: str, actual_transcription: str = "",
                     flagged_words: str = "", language: str = "en") -> int:
        """
        Save a practice session.
        
        Returns:
            Session ID
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                INSERT INTO practice_sessions
                (user_id, target_sentence, actual_transcription, score, grade, flagged_words, language)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, target_sentence, actual_transcription, score, grade, flagged_words, language))
            
            conn.commit()
            session_id = cursor.lastrowid

            # Update user progress
            self._update_user_progress(conn, user_id, score, language)
            
            # Update word stats
            self._update_word_stats(conn, user_id, target_sentence, flagged_words, language)

            return session_id
        except Exception as exc:
            logger.error(f"Failed to save session: {exc}")
            conn.rollback()
            return -1
        finally:
            conn.close()

    def _update_user_progress(self, conn: sqlite3.Connection, user_id: str, new_score: int, language: str):
        """Update aggregated user progress stats."""
        try:
            # Get current stats
            row = conn.execute(
                "SELECT total_sessions, avg_score FROM user_progress WHERE user_id = ?",
                (user_id,)
            ).fetchone()

            if row:
                total = row["total_sessions"] + 1
                avg = ((row["avg_score"] * row["total_sessions"]) + new_score) / total
                conn.execute("""
                    UPDATE user_progress
                    SET total_sessions = ?, avg_score = ?,
                        last_practice = DATE('now'), updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (total, round(avg, 2), user_id))
            else:
                # First session
                conn.execute("""
                    INSERT INTO user_progress
                    (user_id, total_sessions, avg_score, last_practice)
                    VALUES (?, 1, ?, DATE('now'))
                """, (user_id, new_score))

            conn.commit()
        except Exception as exc:
            logger.error(f"Failed to update user progress: {exc}")
            conn.rollback()

    def _update_word_stats(self, conn: sqlite3.Connection, user_id: str,
                           target_sentence: str, flagged_words: str, language: str):
        """Update per-word performance stats."""
        try:
            flagged_set = set()
            if flagged_words:
                # Parse flagged words (simplified - in real usage would be JSON)
                flagged_set = {w.strip().lower() for w in flagged_words.split(",") if w.strip()}

            words = target_sentence.lower().split()
            for word in words:
                word = word.strip(".,!?;:'\"")
                is_correct = word not in flagged_set
                
                conn.execute("""
                    INSERT INTO user_word_stats
                    (user_id, word, language, times_attempted, times_correct, last_attempted)
                    VALUES (?, ?, ?, 1, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id, word, language) DO UPDATE SET
                        times_attempted = times_attempted + 1,
                        times_correct = times_correct + ?,
                        success_rate = CAST(times_correct + ? AS REAL) / (times_attempted + 1),
                        last_attempted = CURRENT_TIMESTAMP
                """, (user_id, word, language, 1 if is_correct else 0,
                      1 if is_correct else 0, 1 if is_correct else 0))

            conn.commit()
        except Exception as exc:
            logger.error(f"Failed to update word stats: {exc}")
            conn.rollback()

    def get_user_progress(self, user_id: str) -> dict[str, Any]:
        """Get user progress stats."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM user_progress WHERE user_id = ?",
                (user_id,)
            ).fetchone()

            if row:
                return dict(row)
            return {}
        finally:
            conn.close()

    def get_recent_sessions(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get user's recent practice sessions."""
        conn = self._get_connection()
        try:
            rows = conn.execute("""
                SELECT * FROM practice_sessions
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit)).fetchall()

            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_sentence(self, level: str = "A1", language: str = "en") -> dict[str, Any] | None:
        """Get a practice sentence for the given level and language."""
        conn = self._get_connection()
        try:
            row = conn.execute("""
                SELECT * FROM sentence_library
                WHERE level = ? AND language = ?
                ORDER BY times_practiced ASC
                LIMIT 1
            """, (level, language)).fetchone()

            if row:
                # Increment times_practiced
                conn.execute(
                    "UPDATE sentence_library SET times_practiced = times_practiced + 1 WHERE id = ?",
                    (row["id"],)
                )
                conn.commit()
                return dict(row)
            return None
        finally:
            conn.close()

    def get_weak_words(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get words with lowest success rate."""
        conn = self._get_connection()
        try:
            rows = conn.execute("""
                SELECT * FROM user_word_stats
                WHERE user_id = ? AND times_attempted >= 2
                ORDER BY success_rate ASC
                LIMIT ?
            """, (user_id, limit)).fetchall()

            return [dict(row) for row in rows]
        finally:
            conn.close()
