import pyaudio
import wave
import numpy as np
import os
import shutil
import logging

logger = logging.getLogger(__name__)

def record_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """
    Record audio from the microphone for a specified duration.
    
    Args:
        duration: Duration to record in seconds.
        sample_rate: The sample rate for recording (default: 16000 Hz).
        
    Returns:
        Numpy array containing the recorded audio data.
        
    Raises:
        OSError: If microphone is not found or permission is denied.
    """
    chunk_size = 1024
    audio_format = pyaudio.paInt16
    channels = 1
    
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=audio_format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)
    except OSError as e:
        logger.error(f"Failed to access microphone: {e}")
        p.terminate()
        raise
        
    logger.info(f"Recording for {duration} seconds...")
    
    frames = []
    
    # Calculate number of chunks needed for the specified duration
    num_chunks = int(sample_rate / chunk_size * duration)
    
    try:
        for _ in range(num_chunks):
            data = stream.read(chunk_size, exception_on_overflow=False)
            # Convert binary data to numpy array
            frames.append(np.frombuffer(data, dtype=np.int16))
    except Exception as e:
        logger.error(f"Error during audio recording: {e}")
        stream.stop_stream()
        stream.close()
        p.terminate()
        raise OSError(f"Recording failed: {e}")
            
    logger.info("Recording complete.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Concatenate all numpy array chunks into one array
    if frames:
        audio_data = np.concatenate(frames)
    else:
        audio_data = np.array([], dtype=np.int16)
        
    return audio_data

def save_recording(audio_data: np.ndarray, filename: str, format_type: str = 'wav') -> str:
    """
    Save the recorded numpy array audio data to a file.
    
    Args:
        audio_data: Numpy array of audio data.
        filename: Output filename or path.
        format_type: Audio format (only 'wav' is supported by this basic function).
        
    Returns:
        The path to the saved audio file.
        
    Raises:
        OSError: If there's an issue with disk space or invalid filename.
        ValueError: If unsupported format.
    """
    if format_type.lower() != 'wav':
        raise ValueError(f"Basic save function only supports 'wav'. For other formats, use pydub.")
        
    output_dir = os.path.dirname(os.path.abspath(filename))
    
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory {output_dir}: {e}")
            raise OSError(f"Invalid filename or path: {filename}")
            
    try:
        # Check disk space (approximation, requires at least the array size + 1MB buffer)
        total, used, free = shutil.disk_usage(output_dir if output_dir else ".")
        if free < audio_data.nbytes + 1024 * 1024:
            raise OSError("Insufficient disk space to save recording.")
            
        sample_rate = 16000 # Default fallback, normally this should align with record_audio
        
        # We need an instance of PyAudio to get sample size, but we can just use 2 for paInt16
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 2 bytes for int16
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
            
        logger.info(f"Audio saved successfully to {filename}")
        return filename
    except OSError as e:
        logger.error(f"OS Error saving recording: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving recording: {e}")
        raise OSError(f"Failed to save recording to {filename}: {e}")
