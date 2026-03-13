"""
Voice Matcher — cosine-similarity engine for comparing speaker embeddings.

Uses Resemblyzer's VoiceEncoder to embed audio segments and compares them
against an enrolled speaker profile via cosine similarity.
"""
import logging
from typing import List, Dict, Any

import numpy as np
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

# Default match threshold — tuned for Resemblyzer GE2E embeddings.
# Values: 0=completely different, 1=identical voice.
DEFAULT_THRESHOLD = 0.75


class VoiceMatcher:
    """
    Compare voice segments against a reference embedding and classify matches.
    """

    def __init__(self, similarity_threshold: float = DEFAULT_THRESHOLD):
        self.threshold = similarity_threshold
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None:
            from resemblyzer import VoiceEncoder
            logger.info("Loading VoiceEncoder for matching…")
            self._encoder = VoiceEncoder()
        return self._encoder

    # ── Core similarity ───────────────────────────────────────────────────

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Return cosine similarity ∈ [0, 1] (1 = identical)."""
        return float(1.0 - cosine(a, b))

    def is_match(self, similarity: float) -> bool:
        return similarity >= self.threshold

    # ── Segment-level matching ────────────────────────────────────────────

    def embed_segment(self, wav_16k: np.ndarray) -> np.ndarray:
        """Embed a raw 16kHz mono numpy array → 256-dim vector."""
        return self.encoder.embed_utterance(wav_16k)

    def match_segments(
        self,
        reference_embedding: np.ndarray,
        diarized_segments: List[Dict[str, Any]],
        audio_wav: np.ndarray,
        sample_rate: int = 16_000,
    ) -> List[Dict[str, Any]]:
        """
        Compare each diarized segment against the reference embedding.

        Args:
            reference_embedding: 256-dim embedding from enrolled profile.
            diarized_segments:   List of {'start', 'end', 'speaker', …}.
            audio_wav:           Full conversation as 16kHz mono np.ndarray.
            sample_rate:         Sample rate of audio_wav (default 16000).

        Returns:
            Enriched segment list with added keys:
                similarity (float), is_match (bool), confidence_pct (float)
        """
        results = []
        for seg in diarized_segments:
            start_s = int(seg["start"] * sample_rate)
            end_s   = int(seg["end"]   * sample_rate)
            chunk   = audio_wav[start_s:end_s]

            # Skip very short chunks — not enough data for a reliable embedding
            if len(chunk) < sample_rate * 0.8:
                seg = {**seg, "similarity": 0.0, "is_match": False, "confidence_pct": 0.0}
                results.append(seg)
                continue

            try:
                emb        = self.embed_segment(chunk)
                sim        = self.cosine_similarity(reference_embedding, emb)
                is_matched = self.is_match(sim)
            except Exception as e:
                logger.warning(f"Embedding failed for segment [{seg['start']:.1f}–{seg['end']:.1f}s]: {e}")
                sim, is_matched = 0.0, False

            results.append({
                **seg,
                "similarity":      round(sim, 4),
                "is_match":        is_matched,
                "confidence_pct":  round(sim * 100, 1),
            })

            logger.debug(
                f"  [{seg['start']:.1f}–{seg['end']:.1f}s] "
                f"sim={sim:.3f} {'✓ MATCH' if is_matched else '✗'}"
            )

        matched = sum(1 for r in results if r["is_match"])
        logger.info(f"Matching done: {matched}/{len(results)} segments match target speaker.")
        return results

    # ── Sliding-window matching (no prior diarization needed) ─────────────

    def sliding_window_match(
        self,
        reference_embedding: np.ndarray,
        audio_wav: np.ndarray,
        window_s: float = 2.0,
        hop_s: float = 1.0,
        sample_rate: int = 16_000,
    ) -> List[Dict[str, Any]]:
        """
        Match reference voice over a sliding window (fallback when no diarizer).

        Returns list of windows with similarity + is_match.
        """
        window = int(window_s * sample_rate)
        hop    = int(hop_s    * sample_rate)
        total  = len(audio_wav)
        results = []

        for start in range(0, total - window, hop):
            chunk = audio_wav[start: start + window]
            try:
                emb = self.embed_segment(chunk)
                sim = self.cosine_similarity(reference_embedding, emb)
            except Exception:
                sim = 0.0

            results.append({
                "start":          round(start / sample_rate, 2),
                "end":            round((start + window) / sample_rate, 2),
                "similarity":     round(sim, 4),
                "is_match":       self.is_match(sim),
                "confidence_pct": round(sim * 100, 1),
                "speaker":        "TARGET" if self.is_match(sim) else "OTHER",
            })

        return results
