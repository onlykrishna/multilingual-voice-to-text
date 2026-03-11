from pydub import AudioSegment
import logging

logger = logging.getLogger(__name__)

def convert_to_wav(audio_data: AudioSegment, target_sample_rate: int = 16000) -> AudioSegment:
    """
    Convert an AudioSegment to a specific format. Since pydub natively
    handles AudioSegments in memory, this function mainly adjusts the 
    frame rate and channels to standard WAV specs used in speech recognition.
    
    Args:
        audio_data: The loaded AudioSegment.
        target_sample_rate: The desired sample rate (default 16000 Hz).
        
    Returns:
        The processed AudioSegment object.
        
    Raises:
        ValueError: If the audio data is invalid.
    """
    if not isinstance(audio_data, AudioSegment):
        logger.error(f"Expected AudioSegment, got {type(audio_data)}")
        raise ValueError("Invalid audio data type. Must be AudioSegment.")
        
    try:
        # Resample to target sample rate
        audio = audio_data.set_frame_rate(target_sample_rate)
        
        # Convert stereo to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
            
        logger.info(f"Converted audio: {target_sample_rate}Hz, 1 channel (mono)")
        return audio
    except Exception as e:
        logger.error(f"Failed to convert audio: {e}")
        raise ValueError(f"Conversion failed: {e}")
