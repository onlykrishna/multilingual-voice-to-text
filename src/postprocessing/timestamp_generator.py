import logging
import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def _format_time_srt(seconds: float) -> str:
    """Format seconds into HH:MM:SS,mmm for SRT."""
    td = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def _format_time_vtt(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm for VTT."""
    td = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int(td.microseconds / 1000)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"

def add_timestamps(transcription_chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Generate timestamped subtitle formats (SRT and VTT) from chunks.
    
    Args:
        transcription_chunks: List of segment dictionaries containing 
                              'text', 'duration_seconds', etc.
                              
    Returns:
        Dict with 'srt' and 'vtt' string contents.
    """
    logger.info("Generating SRT and VTT formats...")
    
    srt_output = []
    vtt_output = ["WEBVTT", ""]
    
    current_time = 0.0
    
    for i, chunk in enumerate(transcription_chunks):
        text = chunk.get('text', '').strip()
        if not text:
            continue
            
        duration = chunk.get('duration_seconds', 0.0)
        start_time = current_time
        end_time = current_time + duration
        
        # SRT format
        srt_output.append(str(i + 1))
        srt_output.append(f"{_format_time_srt(start_time)} --> {_format_time_srt(end_time)}")
        srt_output.append(text)
        srt_output.append("")  # Blank line
        
        # VTT format
        vtt_output.append(f"{_format_time_vtt(start_time)} --> {_format_time_vtt(end_time)}")
        vtt_output.append(text)
        vtt_output.append("")  # Blank line
        
        current_time = end_time
        
    return {
        "srt": "\n".join(srt_output).strip(),
        "vtt": "\n".join(vtt_output).strip()
    }
