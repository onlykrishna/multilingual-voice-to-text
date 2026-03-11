import logging
import datetime
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

def generate_report(transcription_data: Dict[str, Any], filepath: str = None) -> str:
    """
    Create a processing summary report.

    Args:
        transcription_data: Metadata containing stats from processing
        filepath: Optional path to save the report to.
        
    Returns:
        The formatted textual report.
    """
    logger.info("Generating processing report...")
    
    total_time = transcription_data.get('total_processing_time', 0.0)
    audio_dur = transcription_data.get('audio_duration', 0.0)
    rtf = (total_time / audio_dur) if audio_dur > 0 else 0.0
    
    report_lines = [
        "========================================",
        "     VOICE-TO-TEXT SYSTEM REPORT        ",
        "========================================",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "--- Processing Statistics ---",
        f"Total Processing Time: {total_time:.2f} seconds",
        f"Audio Duration: {audio_dur:.2f} seconds",
        f"Real-Time Factor (RTF): {rtf:.2f}x",
        f"Number of Segments: {transcription_data.get('segment_count', 0)}",
        "",
        "--- Output Quality ---",
        f"Detected Language: {transcription_data.get('language', 'Unknown')}",
        f"Average Confidence: {(transcription_data.get('confidence', 0.0) * 100):.1f}%",
        f"Word Count: {len(transcription_data.get('transcription', '').split())}",
        "========================================"
    ]
    
    report_text = "\n".join(report_lines)
    
    if filepath:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Saved report to {filepath}")
        except IOError as e:
            logger.error(f"Failed to write report to file: {e}")
            
    return report_text
