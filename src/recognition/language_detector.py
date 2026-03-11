"""
Language detection helper — used only for Google engine with auto-detect.
Whisper has native language detection so this is bypassed in that case.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Maps short lang codes → Google's BCP-47 codes
LANG_CODE_MAP = {
    'en': 'en-US', 'hi': 'hi-IN', 'es': 'es-ES', 'fr': 'fr-FR',
    'de': 'de-DE', 'it': 'it-IT', 'pt': 'pt-BR', 'ru': 'ru-RU',
    'ja': 'ja-JP', 'ko': 'ko-KR', 'zh': 'zh-CN', 'ar': 'ar-SA',
    'tr': 'tr-TR', 'nl': 'nl-NL', 'pl': 'pl-PL', 'sv': 'sv-SE',
    'da': 'da-DK', 'fi': 'fi-FI', 'uk': 'uk-UA', 'cs': 'cs-CZ',
}

def detect_language(audio_data, fallback: str = 'en-US') -> str:
    """
    Try to detect language by running a quick Google transcription and
    inferring language from the output text using langdetect.

    Falls back to `fallback` (default 'en-US') on any error.

    NOTE: For accurate multilingual detection, use Whisper engine which
    auto-detects language natively and far more reliably.
    """
    import speech_recognition as sr
    from io import BytesIO

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300

    try:
        # Use first 8 seconds for detection speed
        sample = audio_data[:8000]
        wav_io = BytesIO()
        sample.set_channels(1).set_frame_rate(16000).export(wav_io, format='wav')
        wav_io.seek(0)

        with sr.AudioFile(wav_io) as source:
            audio = recognizer.record(source)

        # Run Google with no language hint to get a raw transcript
        text = recognizer.recognize_google(audio, show_all=False)

        if text and len(text.strip()) > 3:
            try:
                from langdetect import detect
                from langdetect import DetectorFactory
                DetectorFactory.seed = 42
                short_code = detect(text)
                bcp47 = LANG_CODE_MAP.get(short_code, f'{short_code}-{short_code.upper()}')
                logger.info(f"Detected language: {short_code} → {bcp47}")
                return bcp47
            except Exception as e:
                logger.warning(f"langdetect failed: {e}")

    except sr.UnknownValueError:
        logger.warning("No speech detected for language detection.")
    except sr.RequestError as e:
        logger.warning(f"Google API unavailable for language detection: {e}")
    except Exception as e:
        logger.error(f"Language detection error: {e}")

    logger.info(f"Language detection falling back to '{fallback}'")
    return fallback
