import re
import logging

logger = logging.getLogger(__name__)

def format_transcription(text: str, add_punctuation: bool = True) -> str:
    """
    Capitalize sentences and fix spacing issues. Option to add basic periods.
    
    Args:
        text: The raw transcription string.
        add_punctuation: Currently basic fallback if model doesn't support punctuation.
        
    Returns:
        Formatted text.
    """
    if not text:
        return text
        
    logger.info("Formatting text...")
    
    # Basic spacing fix
    formatted_text = re.sub(r'\s+', ' ', text).strip()
    
    if add_punctuation and len(formatted_text) > 0:
        # Capitalize first letter
        formatted_text = formatted_text[0].upper() + formatted_text[1:]
        
        # Add period if not present
        if not formatted_text.endswith(('.', '!', '?')):
            formatted_text += '.'
            
    # Remove simple filler words (optional)
    fillers = [r'\bum\b', r'\buh\b', r'\bhm\b', r'\boh\b', r'\blah\b', r'\blike\b']
    for filler in fillers:
        # Replaces fillers with a space, then we clean up double spaces
        formatted_text = re.sub(filler, '', formatted_text, flags=re.IGNORECASE)
        
    formatted_text = re.sub(r'\s+', ' ', formatted_text).strip()
    
    logger.info("Text formatting complete.")
    return formatted_text
