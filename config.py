# Configuration settings for voice-to-text system

# Audio Input Settings
MICROPHONE_SAMPLE_RATE = 16000  # Hz
MICROPHONE_CHUNK_SIZE = 1024
MICROPHONE_CHANNELS = 1  # Mono

# Preprocessing Settings
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
NOISE_THRESHOLD = -40  # dB
SILENCE_THRESHOLD = -40  # dB
MIN_SILENCE_LENGTH = 500  # milliseconds
SEGMENT_MAX_LENGTH = 60  # seconds

# Recognition Settings
DEFAULT_RECOGNIZER = 'whisper'  # Options: 'google', 'whisper' — Whisper is more accurate
DEFAULT_LANGUAGE = 'auto'  # Auto-detect (Whisper detects natively) or specify e.g. 'en-US'
RECOGNITION_TIMEOUT = 10  # seconds
PHRASE_TIME_LIMIT = 30  # seconds
MAX_RETRIES = 3

# Whisper Model Settings (if using Whisper)
WHISPER_MODEL_SIZE = 'small'  # Options: 'tiny'(fastest), 'base', 'small'(best balance), 'medium', 'large'(most accurate)

# Output Settings
OUTPUT_DIRECTORY = './data/output/'
SAVE_FORMATS = ['txt', 'json', 'srt']
INCLUDE_TIMESTAMPS = True
INCLUDE_CONFIDENCE_SCORES = True

# Logging Settings
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = './logs/voice_to_text.log'
