"""Echo TTS — generates correct pronunciation audio using Kokoro."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class EchoTTS:
    """
    Text-to-speech using existing Metis Kokoro setup.
    Generates reference pronunciation audio with caching.
    """

    def __init__(
        self,
        language: str = "en",
        output_dir: str | None = None,
    ):
        self.language = language
        self.output_dir = output_dir or os.path.expanduser("~/.metis/echo-tts")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def generate(self, text: str) -> str:
        """
        Generate TTS audio for given text.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Path to generated .wav file, or empty string on failure
        """
        # Use hash of text for caching
        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        filename = f"echo_{self.language}_{text_hash}.wav"
        output_path = Path(self.output_dir) / filename

        # Check cache first
        if output_path.exists():
            logger.info(f"EchoTTS cache hit: {output_path}")
            return str(output_path)

        try:
            # Use existing Metis TTS function
            from src.tts import synthesize

            temp_path = synthesize(text, lang=self.language)

            if temp_path and Path(temp_path).exists():
                # Move to our cache directory
                shutil.move(temp_path, output_path)
                logger.info(f"EchoTTS generated: {output_path}")
                return str(output_path)
            else:
                logger.warning("EchoTTS synthesize returned empty path")
                return ""

        except Exception as exc:
            logger.exception(f"EchoTTS generation failed: {exc}")
            return ""
