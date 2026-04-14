"""TTS voice module for Metis — Kokoro + Piper with ES/EN/ZH support."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class VoiceEngine(Enum):
    KOKORO = "kokoro"
    PIPER = "piper"


# Voice configs: (lang, speaker_id, sample_rate, description)
KOKORO_VOICES = {
    "en": {"voice": "af_heart", "sr": 24000, "desc": "Female, clear English"},
    "es": {"voice": "ef_dora", "sr": 24000, "desc": "Female, clear Spanish"},
    "zh": {"voice": "zf_xiaoxiao", "sr": 24000, "desc": "Female, Mandarin"},
}

PIPER_VOICES = {
    "en": {
        "model": os.path.expanduser(
            "~/.cache/huggingface/hub/models--rhasspy--piper-voices/snapshots/7a6c333ec560f0e688371adc2fbb7bbe105028c6/en/en_US/ryan/high/en_US-ryan-high.onnx"
        ),
        "desc": "Male, English",
    },
    "es": {
        "model": os.path.expanduser(
            "~/.cache/huggingface/hub/models--rhasspy--piper-voices/snapshots/7a6c333ec560f0e688371adc2fbb7bbe105028c6/es/es_MX/ald/medium/es_MX-ald-medium.onnx"
        ),
        "desc": "Male, Mexican Spanish",
    },
}


def _setup_espeak():
    """Setup espeak-ng paths for Kokoro (must run before import)."""
    try:
        import espeakng_loader
        import espeakng

        espeak_data_path = espeakng_loader.get_data_path()
        if espeak_data_path:
            wrapper = espeakng.EspeakWrapper()
            wrapper.data_path = espeak_data_path
            library_path = espeakng_loader.get_library_path()
            if library_path:
                wrapper.library_path = library_path
            logger.info("espeak-ng configured: data=%s lib=%s", espeak_data_path, library_path)
            return True
    except Exception as e:
        logger.warning("espeak setup failed: %s", e)
    return False


def synthesize_kokoro(text: str, lang: str = "en") -> str | None:
    """Synthesize speech using Kokoro (PyTorch, CPU). Returns wav path."""
    voice_cfg = KOKORO_VOICES.get(lang)
    if not voice_cfg:
        logger.error("No Kokoro voice for lang: %s", lang)
        return None

    _setup_espeak()

    try:
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code=voice_cfg["voice"][:2])
        voice_name = voice_cfg["voice"]

        out_path = os.path.join(
            tempfile.gettempdir(), f"metis_tts_{int(time.time())}.wav"
        )

        # Kokoro generates audio from text
        generator = pipeline(text, voice=voice_name)
        # Collect all audio segments and concatenate
        import torch

        audio_segments = []
        for result in generator:
            audio_segments.append(result.audio)

        if not audio_segments:
            logger.error("Kokoro produced no audio for: %s", text[:50])
            return None

        audio = torch.cat(audio_segments)

        # Save as WAV
        import torchaudio

        torchaudio.save(
            out_path,
            audio.unsqueeze(0),
            sample_rate=voice_cfg["sr"],
        )

        logger.info("Kokoro TTS → %s (%d samples)", out_path, len(audio))
        return out_path

    except Exception as e:
        logger.error("Kokoro TTS failed: %s", e)
        return None


def synthesize_piper(text: str, lang: str = "en") -> str | None:
    """Synthesize speech using Piper TTS (ultralight, CPU). Returns wav path."""
    voice_cfg = PIPER_VOICES.get(lang)
    if not voice_cfg:
        logger.error("No Piper voice for lang: %s", lang)
        return None

    model_path = voice_cfg["model"]

    if not os.path.exists(model_path):
        logger.error("Piper model not found: %s", model_path)
        return None

    out_path = os.path.join(
        tempfile.gettempdir(), f"metis_tts_{int(time.time())}.wav"
    )

    try:
        import subprocess

        result = subprocess.run(
            ["piper", "-m", model_path, "-f", out_path],
            input=text,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0 and os.path.exists(out_path):
            logger.info("Piper TTS → %s", out_path)
            return out_path
        else:
            logger.error("Piper failed: %s", result.stderr[:200])
            return None

    except Exception as e:
        logger.error("Piper TTS failed: %s", e)
        return None


def synthesize(
    text: str,
    lang: str = "en",
    engine: VoiceEngine = VoiceEngine.KOKORO,
) -> str | None:
    """Synthesize speech with fallback. Kokoro first (female voices), Piper fallback."""
    if engine == VoiceEngine.PIPER:
        result = synthesize_piper(text, lang)
        if result:
            return result
        logger.info("Piper failed, falling back to Kokoro")
        return synthesize_kokoro(text, lang)
    else:
        result = synthesize_kokoro(text, lang)
        if result:
            return result
        logger.info("Kokoro failed, falling back to Piper")
        return synthesize_piper(text, lang)
