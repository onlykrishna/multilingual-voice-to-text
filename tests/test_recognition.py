import pytest
from pydub import AudioSegment
from src.recognition.speech_recognizer import SpeechRecognizer
from src.recognition.language_detector import detect_language

def test_speech_recognizer_init():
    """Test object initialization defaults."""
    sr = SpeechRecognizer(engine='google', language='en-US')
    assert sr.engine == 'google'
    assert sr.default_language == 'en-US'

def test_language_detection_fallback(monkeypatch):
    """Test the language detection handles errors gracefully and falls back to en-US."""
    # Mocking detect_language to simulate an error and fallback
    audio = AudioSegment.silent(duration=500, frame_rate=16000)
    
    # Normally this would be tested with real speech or mocked properly
    # For now, we test the fallback logic
    lang = detect_language(audio)
    assert isinstance(lang, str)
    assert lang == 'en-US' # Should fall back to en-US since there is no speech

def test_batch_transcription_empty():
    """Test transcription holds up on empty list."""
    sr = SpeechRecognizer(engine='google')
    
    res = sr.batch_transcribe([])
    assert res['transcription'] == ''
    assert res['confidence'] == 0.0
    assert len(res['segments']) == 0
