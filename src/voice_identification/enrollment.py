"""
Voice Enrollment Module — process a sample voice recording and create
a speaker profile (embedding vector) that can be used for later matching.

Primary engine: Resemblyzer (GE2E speaker embeddings, 256-dim)
No internet required, works fully offline.
"""
import os
import uuid
import pickle
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class VoiceEnrollment:
    """
    Handles voice sample processing and speaker profile creation.

    Responsibilities:
    - Validate audio quality (duration, SNR, silence ratio)
    - Extract 256-dimensional speaker embedding via Resemblyzer
    - Persist the profile as a pickle on disk
    - Load and delete existing profiles
    """

    def __init__(self, profiles_dir: str = "models/voice_profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._encoder = None   # lazy-loaded

    @property
    def encoder(self):
        """Lazy-load the Resemblyzer VoiceEncoder."""
        if self._encoder is None:
            from resemblyzer import VoiceEncoder
            logger.info("Loading Resemblyzer VoiceEncoder…")
            self._encoder = VoiceEncoder()
            logger.info("VoiceEncoder ready.")
        return self._encoder

    # ── Validation ────────────────────────────────────────────────────────

    def validate_sample(self, audio_path: str) -> Dict[str, Any]:
        """
        Validate voice sample quality.

        Args:
            audio_path: Path to the sample audio file.

        Returns:
            dict with keys:
                is_valid (bool), duration (float), quality_score (int 0-100),
                snr (float), silence_ratio (float), issues (list[str])
        """
        from resemblyzer import preprocess_wav

        issues = []

        try:
            wav = preprocess_wav(audio_path)   # resampled to 16 kHz, mono
        except Exception as e:
            return {
                "is_valid": False, "duration": 0, "quality_score": 0,
                "snr": 0, "silence_ratio": 1, "issues": [f"Cannot load audio: {e}"]
            }

        duration = len(wav) / 16_000

        if duration < 3.0:
            issues.append(f"Too short ({duration:.1f}s — minimum 3s required)")
        if duration > 60.0:
            issues.append(f"Too long ({duration:.1f}s — maximum 60s)")

        # Silence ratio (samples near zero)
        silence_ratio = float(np.mean(np.abs(wav) < 0.01))
        if silence_ratio > 0.5:
            issues.append(f"Too much silence ({silence_ratio*100:.0f}%)")

        # Simplified SNR
        signal_power = float(np.mean(wav ** 2))
        noise_floor  = float(np.percentile(np.abs(wav), 10) ** 2)
        snr = 10 * np.log10(signal_power / (noise_floor + 1e-10))

        if snr < 10:
            issues.append(f"Very noisy audio (SNR ≈ {snr:.1f} dB — recommend ≥15 dB)")
        elif snr < 15:
            issues.append(f"Moderate noise (SNR ≈ {snr:.1f} dB)")

        # Quality score (starts at 100, penalised per issue)
        quality = 100 - len(issues) * 12 - max(0, (20 - snr) * 1.5)
        quality = max(0, min(100, int(quality)))

        return {
            "is_valid": quality > 40 and duration >= 3.0,
            "duration": round(duration, 2),
            "quality_score": quality,
            "snr": round(snr, 1),
            "silence_ratio": round(silence_ratio, 3),
            "issues": issues,
        }

    # ── Enrollment ────────────────────────────────────────────────────────

    def create_profile(self, audio_path: str, speaker_name: str = None) -> Dict[str, Any]:
        """
        Extract a voice embedding and save a speaker profile.

        Args:
            audio_path:   Path to sample voice audio.
            speaker_name: Human-readable label (optional).

        Returns:
            Profile dict: speaker_id, embedding (np.ndarray), metadata.

        Raises:
            ValueError: If validation fails hard.
        """
        from resemblyzer import preprocess_wav
        import datetime

        validation = self.validate_sample(audio_path)
        logger.info(f"Validation: quality={validation['quality_score']}, issues={validation['issues']}")

        if not validation["is_valid"]:
            raise ValueError(f"Voice sample rejected: {'; '.join(validation['issues'])}")

        wav = preprocess_wav(audio_path)
        embedding = self.encoder.embed_utterance(wav)   # shape (256,)

        speaker_id = f"spk_{uuid.uuid4().hex[:8]}"
        timestamp  = datetime.datetime.now().isoformat()

        profile = {
            "speaker_id": speaker_id,
            "embedding":  embedding,
            "metadata": {
                "name":             speaker_name or f"Speaker_{speaker_id[-4:]}",
                "created_at":       timestamp,
                "sample_duration":  validation["duration"],
                "quality_score":    validation["quality_score"],
                "snr":              validation["snr"],
                "silence_ratio":    validation["silence_ratio"],
                "embedding_size":   len(embedding),
                "source_file":      os.path.basename(audio_path),
            }
        }

        path = self.profiles_dir / f"{speaker_id}.pkl"
        with open(path, "wb") as f:
            pickle.dump(profile, f)

        logger.info(f"Profile saved: {path}")
        return profile

    # ── CRUD ──────────────────────────────────────────────────────────────

    def load_profile(self, speaker_id: str) -> Dict[str, Any]:
        """Load an existing speaker profile from disk."""
        path = self.profiles_dir / f"{speaker_id}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Profile not found: {speaker_id}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def list_profiles(self) -> list:
        """Return a list of all stored speaker profile metadata."""
        profiles = []
        for pkl in sorted(self.profiles_dir.glob("*.pkl")):
            try:
                with open(pkl, "rb") as f:
                    p = pickle.load(f)
                profiles.append({
                    "speaker_id":    p["speaker_id"],
                    "name":          p["metadata"]["name"],
                    "created_at":    p["metadata"]["created_at"],
                    "sample_duration": p["metadata"]["sample_duration"],
                    "quality_score": p["metadata"]["quality_score"],
                })
            except Exception:
                pass
        return profiles

    def delete_profile(self, speaker_id: str) -> bool:
        """Delete a speaker profile. Returns True if deleted."""
        path = self.profiles_dir / f"{speaker_id}.pkl"
        if path.exists():
            path.unlink()
            logger.info(f"Deleted profile: {speaker_id}")
            return True
        return False
