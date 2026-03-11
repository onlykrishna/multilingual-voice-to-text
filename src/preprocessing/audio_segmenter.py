from pydub import AudioSegment
from pydub.silence import split_on_silence
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def segment_by_silence(audio_data: AudioSegment, min_silence_len: int = 500, silence_thresh: int = -40) -> List[Dict[str, any]]:
    """
    Split an audio signal by detecting silent portions.
    Attempts to create chunks of manageable length (30-60s) where possible 
    by splitting the audio at natural pauses (silences).
    
    Args:
        audio_data: The audio to segment.
        min_silence_len: Minimum silence length in ms to trigger a split.
        silence_thresh: Minimum dBFS threshold below which is considered silence.
        
    Returns:
        A list of dictionaries containing individual segments and their 
        initial sequential index (timestamps can be computed downstream).
        Example: [{'index': 0, 'chunk': AudioSegment}, ...]
    """
    try:
        logger.info(f"Segmenting audio with {min_silence_len}ms min silence and {silence_thresh}dB threshold.")
        
        # split_on_silence returns a list of AudioSegments
        chunks = split_on_silence(
            audio_data,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=200 # Keep 200ms of silence padding to maintain context
        )
        
        segments = []
        
        for i, chunk in enumerate(chunks):
            segments.append({
                "index": i,
                "chunk": chunk
            })
            
        logger.info(f"Created {len(segments)} audio segments.")
        return segments
        
    except Exception as e:
        logger.error(f"Failed to segment audio: {e}")
        # Return the whole audio as a single chunk on failure
        return [{"index": 0, "chunk": audio_data}]
