from pydub import AudioSegment
import logging

logger = logging.getLogger(__name__)

def reduce_noise(audio_data: AudioSegment, noise_threshold: int = -40) -> AudioSegment:
    """
    Apply basic noise reduction using pydub's built-in filters.
    Preserves typical speech frequencies (300–3400 Hz).
    Uses scipy-based filters when scipy is available, otherwise falls back
    to a pure-pydub volume normalization approach.

    Args:
        audio_data: The audio to process.
        noise_threshold: The silence or noise floor in dBFS (default -40).

    Returns:
        The processed AudioSegment.
    """
    try:
        # Try scipy-based frequency filters if scipy is installed
        from pydub.scipy_effects import high_pass_filter, low_pass_filter
        logger.info("Applying high-pass filter (300Hz) and low-pass filter (3400Hz)")
        audio = high_pass_filter(audio_data, 300)
        audio = low_pass_filter(audio, 3400)
        return audio

    except ImportError:
        # Fallback: normalize volume (no frequency filtering without scipy)
        logger.warning("scipy not found — skipping frequency filtering, applying volume normalization only.")
        target_dBFS = -20.0
        change_in_dBFS = target_dBFS - audio_data.dBFS
        return audio_data.apply_gain(change_in_dBFS)

    except Exception as e:
        logger.error(f"Failed to reduce noise: {e}")
        return audio_data
