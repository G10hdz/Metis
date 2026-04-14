"""Whisper STT engine — faster-whisper transcription with GPU support."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WhisperSTT:
    """
    Speech-to-text engine using faster-whisper.
    Supports CUDA (ROCm) and CPU fallback.
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "auto",
        compute_type: str = "auto",
    ):
        self.model_size = model_size
        self._model = None
        self._device = device
        self._compute_type = compute_type

    def _load_model(self):
        """Lazy-load the Whisper model."""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        # Auto-detect device
        if self._device == "auto":
            # Try CUDA first (ROCm on AMD), fallback to CPU
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16"
                    logger.info("WhisperSTT using CUDA device")
                else:
                    device = "cpu"
                    compute_type = "int8"
                    logger.info("WhisperSTT using CPU device")
            except ImportError:
                device = "cpu"
                compute_type = "int8"
                logger.info("WhisperSTT using CPU (torch not available)")
        else:
            device = self._device
            compute_type = self._compute_type if self._compute_type != "auto" else ("float16" if device == "cuda" else "int8")

        logger.info(f"Loading Whisper model '{self.model_size}' on {device} ({compute_type})")
        
        try:
            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
                download_root=os.path.expanduser("~/.cache/whisper-models"),
            )
            logger.info(f"Whisper model loaded: {self.model_size}")
        except Exception as exc:
            logger.warning(f"Failed to load on {device}, falling back to CPU: {exc}")
            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                download_root=os.path.expanduser("~/.cache/whisper-models"),
            )

    def transcribe(
        self,
        audio_path: str | Path,
        language: str = "en",
        beam_size: int = 5,
    ) -> dict[str, Any]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (.ogg, .wav, .mp3, etc.)
            language: Language code ("en", "es", etc.)
            beam_size: Beam size for decoding (higher = more accurate but slower)
            
        Returns:
            {
                "text": str,
                "language": str,
                "segments": [...],
                "error": str | None
            }
        """
        try:
            self._load_model()
            
            segments, info = self._model.transcribe(
                str(audio_path),
                beam_size=beam_size,
                language=language,
                vad_filter=False,  # We'll use Silero VAD separately if needed
            )

            # Collect full text
            full_text = []
            segments_list = []
            
            for segment in segments:
                segments_list.append({
                    "text": segment.text.strip(),
                    "start": segment.start,
                    "end": segment.end,
                })
                full_text.append(segment.text.strip())

            result_text = " ".join(full_text).strip()
            
            logger.info(f"WhisperSTT transcription complete: {len(result_text)} chars, language={info.language}")
            
            return {
                "text": result_text,
                "language": info.language,
                "segments": segments_list,
                "error": None,
            }
            
        except Exception as exc:
            logger.exception(f"WhisperSTT transcription failed: {exc}")
            return {
                "text": "",
                "language": language,
                "segments": [],
                "error": str(exc),
            }

    def transcribe_telegram_voice(self, voice_file_path: str, language: str = "en") -> dict[str, Any]:
        """
        Convenience method for Telegram voice messages.
        Telegram sends voice as .ogg files with Opus codec.
        
        Args:
            voice_file_path: Path to Telegram voice file
            language: Language code
            
        Returns:
            Same as transcribe()
        """
        return self.transcribe(voice_file_path, language=language)
