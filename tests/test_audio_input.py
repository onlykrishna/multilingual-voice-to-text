import pytest
import numpy as np
from pydub import AudioSegment
from src.audio_input.microphone_capture import record_audio, save_recording
from src.audio_input.file_loader import validate_audio
import os

def test_microphone_capture_returns_numpy_array(monkeypatch):
    """Test that record_audio returns a numpy array."""
    # Since we can't easily record from a real mic in a CI test, we mock it.
    def mock_record(*args, **kwargs):
        return np.zeros(16000, dtype=np.int16)
        
    monkeypatch.setattr('src.audio_input.microphone_capture.record_audio', mock_record)
    
    data = mock_record(1)
    assert isinstance(data, np.ndarray)
    assert len(data) == 16000

def test_validate_audio_low_sample_rate():
    """Test the validation function flags low sample rates."""
    # Create empty quiet AudioSegment at 4000Hz
    audio = AudioSegment.silent(duration=1000, frame_rate=4000)
    
    report = validate_audio(audio)
    assert report['is_valid'] == False
    assert len(report['warnings']) > 0
    assert "Low sample rate" in report['warnings'][0]

def test_validate_audio_valid_format():
    """Test a perfectly valid audio format."""
    audio = AudioSegment.silent(duration=1000, frame_rate=16000)
    audio = audio.set_channels(1)
    
    report = validate_audio(audio)
    assert report['is_valid'] == True
    assert len(report['warnings']) == 0
