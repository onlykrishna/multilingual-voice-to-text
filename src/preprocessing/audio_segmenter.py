from pydub import AudioSegment
from pydub.silence import split_on_silence
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Minimum segment duration in ms — chunks shorter than this get merged
MIN_CHUNK_MS = 1500


def segment_by_silence(
    audio_data: AudioSegment,
    min_silence_len: int = 800,
    silence_thresh: int = -35
) -> List[Dict[str, Any]]:
    """
    Split an audio signal by silent portions, then merge any tiny fragments
    that are shorter than MIN_CHUNK_MS (1.5 s) into their neighbour so that
    every yielded segment is long enough for the STT engine to work with.

    Args:
        audio_data:     The audio to segment.
        min_silence_len: Minimum silence in ms to trigger a split (default 800).
        silence_thresh:  dBFS threshold below which is considered silence (default -35).

    Returns:
        List of dicts:  [{'index': int, 'chunk': AudioSegment}, …]
    """
    logger.info(
        f"Segmenting audio — min_silence={min_silence_len}ms, "
        f"thresh={silence_thresh}dB, audio_len={len(audio_data)/1000:.1f}s"
    )

    try:
        raw_chunks = split_on_silence(
            audio_data,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=300,   # keep 300ms of context padding
        )
    except Exception as e:
        logger.error(f"split_on_silence failed: {e}")
        raw_chunks = [audio_data]

    # If no splits happened (e.g. constant audio), return the whole thing
    if not raw_chunks:
        logger.warning("No chunks produced — returning audio as single segment.")
        return [{"index": 0, "chunk": audio_data}]

    # ── Merge tiny chunks ───────────────────────────────────────────────────
    merged: List[AudioSegment] = []
    buffer: AudioSegment = raw_chunks[0]

    for chunk in raw_chunks[1:]:
        if len(buffer) < MIN_CHUNK_MS:
            # Current buffer too short — absorb the next chunk into it
            buffer = buffer + chunk
        else:
            merged.append(buffer)
            buffer = chunk

    merged.append(buffer)   # flush the last buffer

    segments = [{"index": i, "chunk": c} for i, c in enumerate(merged)]

    logger.info(
        f"Segmentation done — {len(raw_chunks)} raw → {len(segments)} merged segments"
    )
    for s in segments:
        logger.debug(f"  Segment {s['index']}: {len(s['chunk'])/1000:.2f}s")

    return segments
