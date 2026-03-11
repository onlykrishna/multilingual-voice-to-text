"""
Integration tests for the full Voice-to-Text pipeline.
Uses a generated sine wave audio (no real speech, for structural validation only).
"""
import pytest
import os
import json
from pydub import AudioSegment
from src.audio_input.file_loader import load_audio_file, validate_audio
from src.preprocessing.format_converter import convert_to_wav
from src.preprocessing.noise_reducer import reduce_noise
from src.preprocessing.audio_segmenter import segment_by_silence
from src.postprocessing.text_formatter import format_transcription
from src.postprocessing.timestamp_generator import add_timestamps
from src.output.file_writer import save_text, save_json, save_srt
from src.output.report_generator import generate_report

SAMPLE_WAV = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_audio', 'sample_tone.wav')

@pytest.fixture(scope="module", autouse=True)
def generate_sample_audio():
    """Generate a sample WAV before the integration tests run."""
    import wave, struct, math
    os.makedirs(os.path.dirname(SAMPLE_WAV), exist_ok=True)
    sample_rate = 16000
    duration = 3
    frequency = 440
    num_samples = sample_rate * duration
    with wave.open(SAMPLE_WAV, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(num_samples):
            value = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            wf.writeframes(struct.pack('<h', value))
    yield
    # Cleanup optional
    # os.remove(SAMPLE_WAV)

class TestFullPipeline:
    """End-to-end integration tests for the preprocessing pipeline."""

    def test_01_load_audio(self):
        """Phase 2: Load audio file."""
        audio = load_audio_file(SAMPLE_WAV)
        assert isinstance(audio, AudioSegment)
        assert len(audio) > 0

    def test_02_validate_audio(self):
        """Phase 2: Validate loaded audio."""
        audio = load_audio_file(SAMPLE_WAV)
        report = validate_audio(audio)
        assert report['is_valid'] == True
        assert report['sample_rate'] == 16000
        assert report['channels'] == 1

    def test_03_format_conversion(self):
        """Phase 3: Convert audio format."""
        audio = load_audio_file(SAMPLE_WAV)
        converted = convert_to_wav(audio, 16000)
        assert converted.frame_rate == 16000
        assert converted.channels == 1

    def test_04_noise_reduction(self):
        """Phase 3: Noise reduction does not break audio."""
        audio = load_audio_file(SAMPLE_WAV)
        converted = convert_to_wav(audio, 16000)
        cleaned = reduce_noise(converted, -40)
        assert isinstance(cleaned, AudioSegment)
        assert len(cleaned) > 0

    def test_05_segmentation(self):
        """Phase 3: Segmentation produces valid segment list."""
        audio = load_audio_file(SAMPLE_WAV)
        converted = convert_to_wav(audio, 16000)
        cleaned = reduce_noise(converted)
        segments = segment_by_silence(cleaned)
        assert isinstance(segments, list)
        # The sine wave has no silence, so it comes as 1 chunk
        assert len(segments) >= 1
        assert 'chunk' in segments[0]
        assert 'index' in segments[0]

    def test_06_text_formatter(self):
        """Phase 5: Text formatter capitalizes and adds punctuation."""
        sample_text = "hello this is a test"
        result = format_transcription(sample_text)
        assert result[0].isupper()
        assert result.endswith('.')

    def test_07_timestamp_generation(self):
        """Phase 5: Timestamp generation returns SRT and VTT strings."""
        fake_segments = [
            {"text": "Hello world", "duration_seconds": 1.5},
            {"text": "This is a test", "duration_seconds": 2.0},
        ]
        result = add_timestamps(fake_segments)
        assert 'srt' in result
        assert 'vtt' in result
        assert '00:00:00,000' in result['srt']
        assert 'WEBVTT' in result['vtt']

    def test_08_save_and_report(self, tmpdir):
        """Phase 6: Save outputs and generate report."""
        txt_path  = os.path.join(tmpdir, "out.txt")
        json_path = os.path.join(tmpdir, "out.json")
        srt_path  = os.path.join(tmpdir, "out.srt")
        rep_path  = os.path.join(tmpdir, "report.txt")

        data = {
            "transcription": "Hello world. This is a test.",
            "language": "en-US",
            "confidence": 0.92,
            "duration": 3.0,
            "timestamp_created": "2026-03-11T22:00:00",
            "segments": [],
            "total_processing_time": 1.5,
            "audio_duration": 3.0,
            "segment_count": 1
        }

        save_text(data["transcription"], txt_path)
        save_json(data, json_path)
        save_srt("1\n00:00:00,000 --> 00:00:01,500\nHello world.", srt_path)
        report = generate_report(data, rep_path)

        assert os.path.exists(txt_path)
        assert os.path.exists(json_path)
        assert os.path.exists(srt_path)
        assert os.path.exists(rep_path)

        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["transcription"] == "Hello world. This is a test."

        assert "Real-Time Factor" in report
        assert "en-US" in report
