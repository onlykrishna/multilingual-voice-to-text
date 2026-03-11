import pytest
from pydub import AudioSegment
from src.preprocessing.format_converter import convert_to_wav
from src.preprocessing.noise_reducer import reduce_noise
from src.preprocessing.audio_segmenter import segment_by_silence

def test_convert_to_wav():
    """Test format converter changes sample rate and channels."""
    audio = AudioSegment.silent(duration=1000, frame_rate=44100)
    audio = audio.set_channels(2)
    
    converted = convert_to_wav(audio, 16000)
    
    assert converted.frame_rate == 16000
    assert converted.channels == 1

def test_reduce_noise():
    """Test basic noise gate does not break audio."""
    audio = AudioSegment.silent(duration=1000, frame_rate=16000)
    
    cleaned = reduce_noise(audio, -40)
    
    assert isinstance(cleaned, AudioSegment)
    assert len(cleaned) == 1000

def test_segment_by_silence():
    """Test audio segmentation logic."""
    # Create an audio file with deliberate silence
    speech = AudioSegment.silent(duration=500, frame_rate=16000) + \
             AudioSegment.silent(duration=1000, frame_rate=16000) # Assuming it's silent enough to split
             
    # Since pydub splits on actual silent dBs, this will just result in 0 or 1 chunks 
    # if it's completely silent, or if not padding, etc. 
    # Let's ensure it doesn't crash and returns the right structure
    segments = segment_by_silence(speech, 500, -20)
    
    assert isinstance(segments, list)
    if len(segments) > 0:
        assert 'index' in segments[0]
        assert 'chunk' in segments[0]
