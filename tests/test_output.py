import pytest
import os
import json
from src.output.file_writer import save_text, save_json, save_srt
from src.output.report_generator import generate_report

def test_save_text(tmpdir):
    """Test saving text to a file."""
    filepath = os.path.join(tmpdir, "test.txt")
    text_content = "Hello world"
    
    save_text(text_content, filepath)
    
    with open(filepath, 'r') as f:
        content = f.read()
    assert content == "Hello world"

def test_save_json(tmpdir):
    """Test saving JSON to a file."""
    filepath = os.path.join(tmpdir, "test.json")
    data = {
        "transcription": "Test Data",
        "confidence": 0.99
    }
    
    save_json(data, filepath)
    
    with open(filepath, 'r') as f:
        loaded = json.load(f)
    assert loaded["transcription"] == "Test Data"
    assert "confidence" in loaded

def test_generate_report(tmpdir):
    """Test report generation returns string and writes to file."""
    filepath = os.path.join(tmpdir, "report.txt")
    data = {
        "transcription": "A sample text.",
        "language": "en",
        "confidence": 0.95,
        "total_processing_time": 5.0,
        "audio_duration": 10.0,
        "segment_count": 3
    }
    
    report = generate_report(data, filepath)
    
    assert "Real-Time Factor (RTF): 0.50x" in report
    assert os.path.exists(filepath)
