"""
Speaker Diarizer — segment a conversation by speaker using Resemblyzer
embeddings + Agglomerative Clustering (fully offline, no HuggingFace token).

Optional upgrade: use pyannote.audio for higher accuracy if a HF token
is available and the user has accepted pyannote model terms.
"""
import logging
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """
    Segment a conversation recording into per-speaker turns.

    Primary method: Resemblyzer sliding-window embeddings
                    + sklearn AgglomerativeClustering
    Optional method: pyannote.audio neural diarization (requires HF token)
    """

    def __init__(
        self,
        window_s: float = 1.5,
        hop_s: float = 0.5,
        min_speakers: int = 1,
        max_speakers: int = 8,
        hf_token: Optional[str] = None,
    ):
        self.window_s     = window_s
        self.hop_s        = hop_s
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.hf_token     = hf_token
        self._encoder     = None

    @property
    def encoder(self):
        if self._encoder is None:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder()
        return self._encoder

    # ── Offline diarization (default) ─────────────────────────────────────

    def _embed_windows(self, wav: np.ndarray, sr: int = 16_000) -> tuple:
        """Slide a window, embed each chunk, return (embeddings, timestamps)."""
        window   = int(self.window_s * sr)
        hop      = int(self.hop_s    * sr)
        total    = len(wav)
        embeddings, times = [], []

        for start in range(0, total - window, hop):
            chunk = wav[start: start + window]
            try:
                emb = self.encoder.embed_utterance(chunk)
                embeddings.append(emb)
                times.append((start / sr, (start + window) / sr))
            except Exception as e:
                logger.warning(f"Embed window failed at {start/sr:.1f}s: {e}")

        return np.array(embeddings), times

    def _cluster(self, embeddings: np.ndarray, n_speakers: int) -> np.ndarray:
        """AgglomerativeClustering on cosine distance."""
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import normalize

        normed = normalize(embeddings, norm="l2")
        model  = AgglomerativeClustering(
            n_clusters=n_speakers,
            metric="cosine",
            linkage="average",
        )
        return model.fit_predict(normed)

    def _estimate_n_speakers(self, embeddings: np.ndarray) -> int:
        """
        Estimate number of speakers using silhouette score over candidate counts.
        Falls back to 2 if sklearn is unavailable or too few windows.
        """
        if len(embeddings) < 4:
            return 1

        try:
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import normalize

            normed = normalize(embeddings, norm="l2")
            best_n, best_score = self.min_speakers, -1

            for n in range(max(2, self.min_speakers), min(self.max_speakers + 1, len(embeddings))):
                labels = self._cluster(embeddings, n)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(normed, labels, metric="cosine")
                if score > best_score:
                    best_score, best_n = score, n

            logger.info(f"Estimated {best_n} speaker(s) (silhouette={best_score:.3f})")
            return best_n

        except Exception as e:
            logger.warning(f"Speaker count estimation failed: {e} — defaulting to 2")
            return 2

    def _labels_to_segments(
        self, labels: np.ndarray, times: list
    ) -> List[Dict[str, Any]]:
        """Convert per-window labels into merged speaker-turn segments."""
        raw = [
            {"start": t[0], "end": t[1], "speaker": f"SPEAKER_{l:02d}"}
            for l, t in zip(labels, times)
        ]

        if not raw:
            return []

        # Merge consecutive same-speaker windows
        merged = [raw[0].copy()]
        for seg in raw[1:]:
            prev = merged[-1]
            if seg["speaker"] == prev["speaker"] and seg["start"] <= prev["end"] + 0.1:
                prev["end"] = seg["end"]
                prev["duration"] = round(prev["end"] - prev["start"], 3)
            else:
                merged.append(seg.copy())

        for s in merged:
            s["duration"] = round(s["end"] - s["start"], 3)

        # Remove very short segments (< 0.5s) — likely artifacts
        merged = [s for s in merged if s["duration"] >= 0.5]
        return merged

    # ── PyAnnote (optional, requires HF token) ────────────────────────────

    def _pyannote_diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """Use pyannote.audio if available and HF token is set."""
        from pyannote.audio import Pipeline
        import torch

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token,
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        diarization = pipeline(
            audio_path,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start":    round(turn.start, 3),
                "end":      round(turn.end,   3),
                "speaker":  speaker,
                "duration": round(turn.end - turn.start, 3),
            })
        return segments

    # ── Public API ────────────────────────────────────────────────────────

    def diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Segment a conversation audio by speaker.

        Tries pyannote.audio first (if HF token available), then falls back
        to the offline Resemblyzer + clustering approach.

        Args:
            audio_path: Path to the conversation audio file.

        Returns:
            Sorted list of segments: [{start, end, speaker, duration}, …]
        """
        # Try pyannote first if token provided
        if self.hf_token:
            try:
                logger.info("Trying pyannote.audio diarization…")
                segs = self._pyannote_diarize(audio_path)
                logger.info(f"pyannote: {len(segs)} segments")
                return sorted(segs, key=lambda x: x["start"])
            except Exception as e:
                logger.warning(f"pyannote failed ({e}), falling back to offline method.")

        # Offline: Resemblyzer + Agglomerative Clustering
        logger.info("Running offline Resemblyzer + clustering diarization…")
        from resemblyzer import preprocess_wav

        wav        = preprocess_wav(audio_path)
        embeddings, times = self._embed_windows(wav)

        if len(embeddings) == 0:
            logger.warning("No embeddings extracted — returning single segment.")
            total = len(wav) / 16_000
            return [{"start": 0.0, "end": total, "speaker": "SPEAKER_00", "duration": total}]

        n_speakers = self._estimate_n_speakers(embeddings)
        labels     = self._cluster(embeddings, n_speakers)
        segments   = self._labels_to_segments(labels, times)

        logger.info(f"Offline diarization: {n_speakers} speaker(s), {len(segments)} segments")
        return sorted(segments, key=lambda x: x["start"])

    def get_wav(self, audio_path: str) -> np.ndarray:
        """Return preprocessed 16kHz mono wav array for a given audio path."""
        from resemblyzer import preprocess_wav
        return preprocess_wav(audio_path)
