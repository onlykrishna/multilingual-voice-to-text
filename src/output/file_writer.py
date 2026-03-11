import json
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

def save_text(text: str, filepath: str) -> str:
    """Save plain text transcription to file."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Saved text results to {filepath}")
        return filepath
    except IOError as e:
        logger.error(f"Failed to save text file: {e}")
        raise

def save_json(data: Dict[str, Any], filepath: str) -> str:
    """Save transcription data as JSON with metadata."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Serialize format
        payload = {
            "transcription": data.get("transcription", ""),
            "language": data.get("language", "auto"),
            "confidence": data.get("confidence", 0.0),
            "duration": data.get("duration", 0.0),
            "timestamp": data.get("timestamp_created", ""),
            "segments": data.get("segments", [])
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
            
        logger.info(f"Saved JSON results to {filepath}")
        return filepath
    except IOError as e:
        logger.error(f"Failed to save JSON file: {e}")
        raise

def save_srt(timestamped_text: str, filepath: str) -> str:
    """Save the SRT formatted text to file."""
    if not filepath.endswith('.srt'):
        filepath += '.srt'
        
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(timestamped_text)
        logger.info(f"Saved SRT results to {filepath}")
        return filepath
    except IOError as e:
        logger.error(f"Failed to save SRT file: {e}")
        raise
