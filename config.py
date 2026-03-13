# Configuration settings for voice-to-text system

# Audio Input Settings
MICROPHONE_SAMPLE_RATE = 16000  # Hz
MICROPHONE_CHUNK_SIZE = 1024
MICROPHONE_CHANNELS = 1  # Mono

# Preprocessing Settings
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
NOISE_THRESHOLD = -35      # dB — slightly less aggressive (was -40)
SILENCE_THRESHOLD = -35   # dB — a bit looser so short pauses don't over-split
MIN_SILENCE_LENGTH = 800   # milliseconds — longer pause needed to split (was 500)
SEGMENT_MAX_LENGTH = 60   # seconds

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

# ── Voice Identification Settings ─────────────────────────────────────────
VOICE_PROFILES_DIR      = 'models/voice_profiles'
SIMILARITY_THRESHOLD    = 0.75    # 0–1 cosine similarity; ↑ = stricter match
MIN_SAMPLE_DURATION     = 3.0     # seconds — minimum enrollment audio
MAX_SAMPLE_DURATION     = 60.0    # seconds — maximum enrollment audio
MIN_SEGMENT_DURATION    = 0.8     # seconds — ignore diarized chunks shorter than this
SEGMENT_PADDING_MS      = 200     # ms silence between concatenated target-speaker chunks

# Speaker Diarization
DIARIZER_WINDOW_S       = 1.5     # sliding window size (seconds)
DIARIZER_HOP_S          = 0.5     # hop between windows (seconds)
MIN_SPEAKERS            = 1       # minimum speakers to consider
MAX_SPEAKERS            = 8       # maximum speakers to consider
HF_TOKEN                = None    # HuggingFace token (optional, for pyannote.audio upgrade)

