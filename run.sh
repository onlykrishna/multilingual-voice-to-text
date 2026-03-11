#!/bin/bash
# Quick launcher for the Voice-to-Text system
# Run from anywhere: bash /Users/aeologic/krishna/voice_to_text_system/run.sh [args]

# Navigate to project folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Run with any passed arguments (default: interactive mic for 10s)
if [ "$#" -eq 0 ]; then
    echo "🎙️  Voice-to-Text System — starting 10-second mic recording..."
    python main.py --mode mic --duration 10
else
    python main.py "$@"
fi
