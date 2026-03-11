from pydub import AudioSegment
import os
import logging

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

def load_audio_file(filepath: str) -> AudioSegment:
    """
    Load an audio file into an AudioSegment object.
    
    Args:
        filepath: Path to the audio file.
        
    Returns:
        AudioSegment object containing the audio data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported or corrupted.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Audio file not found: {filepath}")
        
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    if ext not in SUPPORTED_FORMATS:
        logger.error(f"Unsupported format: {ext}")
        raise ValueError(f"Unsupported format: {ext}. Supported formats are {SUPPORTED_FORMATS}")
        
    try:
        # Determine the correct format naming for pydub
        format_name = ext[1:] # e.g., 'wav' from '.wav'
        if ext == '.m4a':
            format_name = 'mp4' # ffmpeg treats m4a as an mp4 container stream usually
            
        audio = AudioSegment.from_file(filepath, format=format_name)
        return audio
    except Exception as e:
        logger.error(f"Failed to load audio file {filepath}: {e}")
        raise ValueError(f"File could not be loaded or is corrupted: {filepath}") from e

def validate_audio(audio_data: AudioSegment) -> dict:
    """
    Validate an AudioSegment against required specifications.
    
    Args:
        audio_data: The loaded AudioSegment.
        
    Returns:
        A dictionary containing the validation rules passed and warnings.
    """
    report = {
        "is_valid": True,
        "warnings": [],
        "sample_rate": audio_data.frame_rate,
        "duration_seconds": len(audio_data) / 1000.0,
        "channels": audio_data.channels
    }
    
    # Check sample rate
    if report["sample_rate"] < 8000:
        report["warnings"].append(f"Low sample rate: {report['sample_rate']}Hz. Expected >= 8kHz.")
        report["is_valid"] = False
        
    # Check duration (warn if > 1 hour)
    if report["duration_seconds"] > 3600:
        report["warnings"].append(f"Duration too long: {report['duration_seconds']}s > 3600s")
        
    # Check channels
    if report["channels"] > 2:
        report["warnings"].append(f"More than 2 channels detected: {report['channels']}")
        
    if report["warnings"]:
        logger.warning(f"Audio validation warnings: {report['warnings']}")
        
    return report
