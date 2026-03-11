"""
Generate a sample WAV audio file (440Hz sine wave tone) for testing.
Run this script once to create data/sample_audio/sample_en.wav
"""
import wave
import struct
import math
import os

def generate_sine_wav(filepath, duration=3, sample_rate=16000, frequency=440):
    """Generate a pure sine wave WAV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    num_samples = sample_rate * duration
    
    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(1)       # mono
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sample_rate)
        
        for i in range(num_samples):
            value = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            wf.writeframes(struct.pack('<h', value))
    
    print(f"✓ Generated test WAV: {filepath}")

if __name__ == "__main__":
    generate_sine_wav("data/sample_audio/sample_tone.wav", duration=3)
    print("Sample audio files generated successfully.")
