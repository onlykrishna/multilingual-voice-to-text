"""
Segment Extractor — concatenate audio chunks for the target speaker
and export as a single WAV file.
"""
import os
import logging
from typing import List, Dict, Any

from pydub import AudioSegment

logger = logging.getLogger(__name__)


class SegmentExtractor:
    """
    Extract and concatenate target-speaker audio segments from a conversation.
    """

    def extract_segments(
        self,
        audio_path: str,
        segments: List[Dict[str, Any]],
        padding_ms: int = 200,
    ) -> AudioSegment:
        """
        Load the original conversation and concatenate target-speaker segments.

        Args:
            audio_path:  Path to the original conversation audio file.
            segments:    List of matched segments [{start, end, …}, …].
            padding_ms:  Ms of silence inserted between segments.

        Returns:
            AudioSegment containing concatenated target speech.
        """
        audio   = AudioSegment.from_file(audio_path)
        silence = AudioSegment.silent(duration=padding_ms)
        result  = AudioSegment.empty()

        for seg in sorted(segments, key=lambda s: s["start"]):
            start_ms = int(seg["start"] * 1000)
            end_ms   = int(seg["end"]   * 1000)

            # Guard against out-of-bounds
            start_ms = max(0, start_ms)
            end_ms   = min(len(audio), end_ms)

            if end_ms <= start_ms:
                continue

            chunk  = audio[start_ms:end_ms]
            result = result + chunk + silence

        logger.info(
            f"Extracted {len(segments)} segment(s) → "
            f"{len(result)/1000:.1f}s total audio"
        )
        return result

    def save_extracted_audio(
        self,
        audio: AudioSegment,
        output_path: str,
        fmt: str = "wav",
    ) -> str:
        """
        Export concatenated audio to a file.

        Args:
            audio:       AudioSegment to export.
            output_path: Destination file path.
            fmt:         Format string (wav, mp3, etc.).

        Returns:
            Absolute path of the saved file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        audio.export(output_path, format=fmt)
        logger.info(f"Saved extracted audio → {output_path}")
        return output_path
