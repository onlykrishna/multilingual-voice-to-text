import speech_recognition as sr
import logging
from typing import List, Dict, Any
from io import BytesIO
from pydub import AudioSegment
import time

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    def __init__(self, engine: str = 'google', language: str = 'auto'):
        """
        Initialize the recognition engine.

        Args:
            engine: 'google' or 'whisper'
            language: ISO language code like 'en-US', 'hi-IN', or 'auto'
        """
        self.engine = engine.lower()
        self.default_language = language
        self.recognizer = sr.Recognizer()
        # Adjust recognizer sensitivity for better accuracy
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

        # Lazy-load Whisper only if selected — avoids numpy/torch errors at import time
        self._whisper_model = None

        logger.info(f"SpeechRecognizer ready — engine={engine}, language={language}")

    @property
    def whisper_model(self):
        """Lazy-load Whisper on first use."""
        if self._whisper_model is None:
            from src.recognition.whisper_engine import WhisperEngine
            import config
            self._whisper_model = WhisperEngine(config.WHISPER_MODEL_SIZE)
        return self._whisper_model

    # ── helpers ──────────────────────────────────────────────────────────────

    def _to_sr_audio(self, chunk: AudioSegment) -> sr.AudioData:
        """Convert pydub AudioSegment → SpeechRecognition AudioData."""
        wav_io = BytesIO()
        chunk.set_channels(1).set_frame_rate(16000).export(wav_io, format="wav")
        wav_io.seek(0)
        with sr.AudioFile(wav_io) as src:
            return self.recognizer.record(src)

    def _google_transcribe(self, sr_audio: sr.AudioData, language: str) -> Dict[str, Any]:
        """Call Google free speech API with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Request full JSON response to get confidence
                resp = self.recognizer.recognize_google(
                    sr_audio,
                    language=language if language != 'auto' else 'en-US',
                    show_all=True
                )
                if resp and 'alternative' in resp:
                    best = resp['alternative'][0]
                    return {
                        "text": best.get('transcript', ''),
                        "confidence": best.get('confidence', 0.85)
                    }
                return {"text": "", "confidence": 0.0}

            except sr.UnknownValueError:
                logger.warning("Google: could not understand audio segment.")
                return {"text": "", "confidence": 0.0}
            except sr.RequestError as e:
                logger.warning(f"Google API error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1.5)
        return {"text": "", "confidence": 0.0}

    def _whisper_transcribe(self, chunk: AudioSegment, language: str) -> Dict[str, Any]:
        """Transcribe using local OpenAI Whisper model."""
        lang_arg = None if language == 'auto' else language.split('-')[0]  # 'en-US' → 'en'
        return self.whisper_model.transcribe(chunk, lang_arg)

    # ── public API ────────────────────────────────────────────────────────────

    def transcribe(self, chunk: AudioSegment, language: str = None) -> Dict[str, Any]:
        """
        Transcribe a single audio chunk.

        Returns:
            dict with keys: text, language, confidence
        """
        lang = language or self.default_language

        try:
            if self.engine == 'whisper':
                result = self._whisper_transcribe(chunk, lang)
                return {
                    "text": result.get("text", ""),
                    "language": result.get("language", lang),
                    "confidence": result.get("confidence", 0.9)
                }
            else:
                # Google engine
                sr_audio = self._to_sr_audio(chunk)
                result = self._google_transcribe(sr_audio, lang)
                return {
                    "text": result["text"],
                    "language": lang,
                    "confidence": result["confidence"]
                }
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return {"text": "", "language": lang, "confidence": 0.0}

    def batch_transcribe(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Transcribe all segments and merge into a final result.

        Args:
            segments: list of {'index': int, 'chunk': AudioSegment}

        Returns:
            dict with transcription, language, confidence, segments
        """
        if not segments:
            return {"transcription": "", "language": self.default_language,
                    "confidence": 0.0, "segments": []}

        logger.info(f"Batch transcribing {len(segments)} segment(s) with {self.engine}…")
        texts, confidences, details = [], [], []
        detected_lang = self.default_language

        for seg in segments:
            idx   = seg['index']
            chunk = seg['chunk']
            logger.info(f"  Segment {idx+1}/{len(segments)} ({len(chunk)/1000:.1f}s)…")

            res = self.transcribe(chunk, self.default_language)
            if res["text"]:
                texts.append(res["text"].strip())
                confidences.append(res["confidence"])
                details.append({
                    "segment_index": idx,
                    "text": res["text"],
                    "confidence": res["confidence"],
                    "duration_seconds": len(chunk) / 1000.0
                })
                if detected_lang in ('auto', None):
                    detected_lang = res["language"]

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        full_text = " ".join(texts)

        logger.info(f"Batch done — {len(texts)} segment(s) with text, avg_conf={avg_conf:.2f}")
        return {
            "transcription": full_text,
            "language": detected_lang,
            "confidence": round(avg_conf, 4),
            "segments": details
        }
