# Multilingual Voice-to-Text System

A robust, multilingual voice-to-text conversion system in Python that can accurately transcribe speech from any language into written text.

## Features
- **Dynamic Input Support**: Transcribe from real-time microphone input or a variety of audio files (WAV, MP3, FLAC, OGG).
- **Audio Preprocessing**: Automatic format conversion, sample rate normalization, noise reduction, and smart chunking using silence detection.
- **Dual Recognition Engines**: Fallback between Google Speech Recognition and OpenAI Whisper.
- **Language Detection**: Automatically detect the spoken language.
- **Multiple Output Formats**: Plain text, JSON with metadata, and timestamped SRT/VTT subtitles.

## Requirements
- Python 3.8+
- Audio libraries may require system-level dependencies depending on your platform:
  - **macOS/Linux**: `brew install ffmpeg portaudio` or `apt-get install ffmpeg portaudio19-dev`

## Quick Start Pipeline

1. **Install Virtual Environment and Dependencies**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Basic Usage (Command Line Interface)**
```bash
# Record from microphone for 10 seconds
python main.py --mode mic --duration 10

# Transcribe an audio file
python main.py --mode file --input path/to/audio.mp3

# Batch process a directory
python main.py --mode batch --input-dir path/to/audio_folder/
```

3. **Check Outputs**
Outputs will be generated in `./data/output/` including `.txt`, `.json`, `.srt`, and optional generation reports.

## Configuration
All configuration settings like Recognition Engines, Default Language, and chunk limits are defined in `config.py`. Adjust them to your specific use case.

## System Architecture Highlights
See `/docs/system_design.md` for a comprehensive overview of the design architecture, pipelines, and class hierarchies.
