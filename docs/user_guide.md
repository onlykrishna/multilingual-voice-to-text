# User Guide

Welcome to the Voice-to-Text System user guide.

## Installation
Ensure you have the latest Python 3+ version and necessary audio tools (like FFMPEG).
Run the setup instructions found in `.readme.md`.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Basic CLI Usage
The system supports multiple usage modes directly from the terminal via `main.py`.

### 1. Transcribe the Microphone
To capture audio directly from your system microphone, pass the duration parameter to establish how long it should wait for your voice clip.
```bash
python main.py --mode mic --duration 10
```

### 2. Transcribe a Local File
Supports WAV, MP3, FLAC, M4A out of the box.
```bash
python main.py --mode file --input /path/to/my_audio.mp3
```

### 3. Change Engine Preference
You can swap between lightweight Google tools or offline local OpenAI Whisper parameters within your `config.py` file! Update the variable `DEFAULT_RECOGNIZER` to either `'google'` or `'whisper'`.

### 4. Specifying Languages
If you find auto-detection unreliable, you can force the language dictionary (e.g. `'es-ES'` for Spanish).
```bash
python main.py --mode file --input speech.wav --language es-ES
```

### 5. Running Batches
Drop all your unsorted audio tracks into one specific input directory and run:
```bash
python main.py --mode batch --input-dir raw_tracks/
```

Outputs will always automatically appear grouped inside `./data/output/`.

## Generated Logs
A timestamped file log will continuously dump errors inside `./logs/voice_to_text.log`. Refer to this log file if any pipeline errors out halfway!
