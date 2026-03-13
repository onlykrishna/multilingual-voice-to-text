"""
Tests for the Voice Identification module.
Mocks the resemblyzer library and audio parsing so tests run extremely fast
and don't fail due to synthetic audio failing internal VAD checks.
"""
import os
import sys
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ── Mocks ────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_resemblyzer():
    with patch('resemblyzer.preprocess_wav') as pw_mock:
        # 10 seconds empty data
        def fake_preprocess(*args, **kwargs):
            return np.ones(160000, dtype=np.float32) * 0.5
        pw_mock.side_effect = fake_preprocess
        yield


@pytest.fixture
def mock_voice_encoder():
    with patch('resemblyzer.VoiceEncoder') as VE:
        instance = MagicMock()
        def fake_embed(*args, **kwargs):
            vec = np.random.rand(256).astype(np.float32)
            return vec / np.linalg.norm(vec)
        instance.embed_utterance.side_effect = fake_embed
        VE.return_value = instance
        yield instance


# ── Enrollment ────────────────────────────────────────────────────────────────

class TestVoiceEnrollment:

    def test_validate_sample_too_short(self, mock_resemblyzer):
        """Samples < 3s should report is_valid=False."""
        from src.voice_identification.enrollment import VoiceEnrollment
        with patch('resemblyzer.preprocess_wav') as pw_mock, \
             patch('pydub.AudioSegment.from_file') as pydub_mock:
            pw_mock.return_value = np.ones(16000, dtype=np.float32) * 0.1 # 1 sec non-zero
            pydub_mock.return_value.duration_seconds = 1.0
            
            enroller = VoiceEnrollment(profiles_dir=tempfile.mkdtemp())
            result   = enroller.validate_sample("fake.wav")
            assert result["is_valid"] is False
            assert result["duration"] == 1.0

    def test_validate_sample_valid(self, mock_resemblyzer):
        """A clean 10s wave should pass validation."""
        from src.voice_identification.enrollment import VoiceEnrollment
        with patch('resemblyzer.preprocess_wav') as pw_mock, \
             patch('pydub.AudioSegment.from_file') as pydub_mock:
            pydub_mock.return_value.duration_seconds = 10.0
            
            # 10 seconds of random noise (looks like speech to snippet)
            pw_mock.return_value = (np.random.rand(160000).astype(np.float32) - 0.5)
            enroller = VoiceEnrollment(profiles_dir=tempfile.mkdtemp())
            result   = enroller.validate_sample("fake.wav")
            assert result["is_valid"] is True
            assert result["duration"] == 10.0
            assert "quality_score" in result

    def test_create_profile_saves_embedding(self, mock_resemblyzer, mock_voice_encoder):
        """create_profile should save a .pkl with a 256-dim embedding."""
        from src.voice_identification.enrollment import VoiceEnrollment
        prof_dir  = tempfile.mkdtemp()
        with patch('resemblyzer.preprocess_wav') as pw_mock:
            pw_mock.return_value = (np.random.rand(160000).astype(np.float32) - 0.5)
            enroller = VoiceEnrollment(profiles_dir=prof_dir)
            profile  = enroller.create_profile("fake.wav", "Test Speaker")
            
            assert profile["speaker_id"].startswith("spk_")
            emb = profile["embedding"]
            assert hasattr(emb, '__len__') and len(emb) == 256
            assert profile["metadata"]["name"] == "Test Speaker"
            
            pkl_path = os.path.join(prof_dir, profile["speaker_id"] + ".pkl")
            assert os.path.exists(pkl_path)

    def test_list_and_delete_profile(self, mock_resemblyzer, mock_voice_encoder):
        """list_profiles and delete_profile should correctly manage CRUD."""
        from src.voice_identification.enrollment import VoiceEnrollment
        prof_dir = tempfile.mkdtemp()
        with patch('resemblyzer.preprocess_wav') as pw_mock:
            pw_mock.return_value = (np.random.rand(160000).astype(np.float32) - 0.5)
            enroller = VoiceEnrollment(profiles_dir=prof_dir)
            p = enroller.create_profile("fake.wav")
            sid = p["speaker_id"]

            profiles = enroller.list_profiles()
            assert any(pr["speaker_id"] == sid for pr in profiles)

            deleted = enroller.delete_profile(sid)
            assert deleted is True
            assert not enroller.list_profiles()


# ── Voice Matcher ─────────────────────────────────────────────────────────────

class TestVoiceMatcher:

    def test_cosine_similarity_identical(self):
        """Identical vectors → similarity = 1.0"""
        from src.voice_identification.voice_matcher import VoiceMatcher
        vm  = VoiceMatcher()
        vec = np.random.rand(256).astype(np.float32)
        vec /= np.linalg.norm(vec)
        assert vm.cosine_similarity(vec, vec) == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors → similarity ≈ 0.0"""
        from src.voice_identification.voice_matcher import VoiceMatcher
        vm = VoiceMatcher()
        a  = np.zeros(256); a[0] = 1.0
        b  = np.zeros(256); b[1] = 1.0
        assert vm.cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_is_match_threshold(self):
        from src.voice_identification.voice_matcher import VoiceMatcher
        vm = VoiceMatcher(similarity_threshold=0.75)
        assert vm.is_match(0.80) is True
        assert vm.is_match(0.70) is False
        assert vm.is_match(0.75) is True

    @patch('resemblyzer.VoiceEncoder')
    def test_match_segments_structure(self, mock_ve_cls):
        """match_segments should return enriched dicts with is_match key."""
        mock_ve = MagicMock()
        mock_ve.embed_utterance.return_value = np.zeros(256)
        mock_ve_cls.return_value = mock_ve

        from src.voice_identification.voice_matcher import VoiceMatcher
        vm      = VoiceMatcher()
        ref_emb = np.zeros(256).astype(np.float32)
        audio   = np.zeros(32000, dtype=np.float32)   # 2s silent audio
        segs    = [{"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00", "duration": 1.5}]
        
        results = vm.match_segments(ref_emb, segs, audio, sample_rate=16000)
        assert len(results) == 1
        assert "is_match"   in results[0]
        assert "similarity" in results[0]
        assert "confidence_pct" in results[0]


# ── Speaker Diarizer ──────────────────────────────────────────────────────────

class TestSpeakerDiarizer:

    @patch('resemblyzer.VoiceEncoder')
    def test_diarize_returns_segments(self, mock_ve_cls):
        """Diarizer should safely return fallback when mocked."""
        mock_ve = MagicMock()
        mock_ve.embed_utterance.return_value = np.ones(256) * 0.1
        mock_ve_cls.return_value = mock_ve

        from src.voice_identification.speaker_diarizer import SpeakerDiarizer
        with patch('resemblyzer.preprocess_wav') as pw_mock:
            pw_mock.return_value = np.zeros(160000) # 10 seconds, multiple segments
            d    = SpeakerDiarizer(window_s=1.5, hop_s=0.5)
            segs = d.diarize("fake.wav")
            assert isinstance(segs, list)
            assert len(segs) >= 1
            assert "start" in segs[0]
            assert "end"   in segs[0]
            assert "speaker" in segs[0]


# ── Segment Extractor ─────────────────────────────────────────────────────────

class TestSegmentExtractor:

    def _make_dummy_wav(self, duration_s):
        from pydub import AudioSegment
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio = AudioSegment.silent(duration=int(duration_s * 1000))
        audio.export(tmp.name, format='wav')
        return tmp.name

    def test_extract_segments_duration(self):
        from src.voice_identification.segment_extractor import SegmentExtractor
        wav_path = self._make_dummy_wav(10.0)
        try:
            extractor = SegmentExtractor()
            segs = [
                {"start": 1.0, "end": 3.0}, # 2s
                {"start": 5.0, "end": 7.0}, # 2s
            ]
            audio = extractor.extract_segments(wav_path, segs, padding_ms=0)
            assert abs(len(audio) / 1000 - 4.0) < 0.3
        finally:
            os.unlink(wav_path)


# ── Analytics Generator ───────────────────────────────────────────────────────

class TestAnalyticsGenerator:

    def _make_mock_data(self):
        enrollment_data = {
            "speaker_id": "spk_test01",
            "embedding":  np.zeros(256),
            "metadata": {
                "name":            "Test Speaker",
                "created_at":      "2026-03-13T00:00:00",
                "sample_duration": 10.0,
                "quality_score":   85,
                "snr":             22.0,
                "embedding_size":  256,
                "source_file":     "test.wav",
            }
        }
        all_segs = [
            {"start": 0.0,  "end": 5.0,  "speaker": "SPEAKER_00", "duration": 5.0,  "is_match": True,  "similarity": 0.90},
            {"start": 6.0,  "end": 10.0, "speaker": "SPEAKER_01", "duration": 4.0,  "is_match": False, "similarity": 0.45},
            {"start": 11.0, "end": 15.0, "speaker": "SPEAKER_00", "duration": 4.0,  "is_match": True,  "similarity": 0.88},
        ]
        matched = [s for s in all_segs if s["is_match"]]
        return enrollment_data, all_segs, matched

    def test_report_structure(self):
        from src.voice_identification.analytics_generator import AnalyticsGenerator
        e, a, m = self._make_mock_data()
        report = AnalyticsGenerator().generate_report(e, 15.0, a, m)
        for key in ["enrollment", "conversation", "target_speaker_analysis",
                    "speaking_segments", "other_speakers", "timeline"]:
            assert key in report, f"Missing key: {key}"

    def test_report_calculations(self):
        from src.voice_identification.analytics_generator import AnalyticsGenerator
        e, a, m = self._make_mock_data()
        report = AnalyticsGenerator().generate_report(e, 15.0, a, m)
        tsa = report["target_speaker_analysis"]
        assert tsa["occurrence_count"] == 2
        assert abs(tsa["total_speaking_time"] - 9.0) < 0.01

    def test_svg_timeline_generated(self):
        from src.voice_identification.analytics_generator import AnalyticsGenerator
        e, a, m = self._make_mock_data()
        report = AnalyticsGenerator().generate_report(e, 15.0, a, m)
        svg    = AnalyticsGenerator().generate_svg_timeline(report["timeline"], 15.0)
        assert svg.strip().startswith("<svg")
        assert "TARGET" in svg

