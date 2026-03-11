"""
recognition package — WhisperEngine is lazy-loaded to avoid
numpy/torch import errors at startup when only Google is needed.
"""
from .speech_recognizer import SpeechRecognizer
from .language_detector import detect_language

__all__ = ['SpeechRecognizer', 'detect_language']
