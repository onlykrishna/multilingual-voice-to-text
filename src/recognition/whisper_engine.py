"""
Whisper-based transcription engine using OpenAI's open-source Whisper model.
Model is loaded lazily — only when first used.
"""
import logging
import os
import tempfile
from typing import Dict, Any, Optional
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class WhisperEngine:
    def __init__(self, model_size: str = 'base'):
        """
        Load the Whisper model.

        Args:
            model_size: 'tiny' | 'base' | 'small' | 'medium' | 'large'
                        Larger = more accurate but slower.
                        'small' is a good balance for most use cases.
        """
        import whisper
        logger.info(f"Loading Whisper '{model_size}' model…")
        self.model = whisper.load_model(model_size)
        logger.info("Whisper model loaded.")

    def transcribe(self, audio: AudioSegment, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper.

        Args:
            audio: pydub AudioSegment
            language: ISO 639-1 code like 'en', 'hi', 'es', or None for auto-detect

        Returns:
            dict with text, language, confidence
        """
        # Export to a temp WAV file — Whisper reads from disk
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            audio.set_channels(1).set_frame_rate(16000).export(tmp.name, format='wav')
            tmp.close()

            opts = {
                "fp16": False,           # CPU-safe
                "task": "transcribe",    # "transcribe" keeps original language; use "translate" for English output
                "verbose": False,
            }
            if language and language != 'auto':
                opts["language"] = language

            result = self.model.transcribe(tmp.name, **opts)

            text      = result.get('text', '').strip()
            lang_out  = result.get('language', language or 'auto')
            segments  = result.get('segments', [])

            # Confidence: 1 - average no_speech_prob across segments
            if segments:
                avg_no_speech = sum(s.get('no_speech_prob', 0.1) for s in segments) / len(segments)
                confidence = max(0.0, round(1.0 - avg_no_speech, 4))
            else:
                confidence = 0.9 if text else 0.0

            logger.info(f"Whisper result: lang={lang_out}, conf={confidence:.2f}, words={len(text.split())}")
            return {"text": text, "language": lang_out, "confidence": confidence}

        except Exception as e:
            logger.error(f"Whisper error: {e}", exc_info=True)
            return {"text": "", "language": language or "auto", "confidence": 0.0}
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)
