import argparse
import sys
import logging
import time
import os
import datetime

import config
from src.audio_input.microphone_capture import record_audio, save_recording
from src.audio_input.file_loader import load_audio_file, validate_audio
from src.preprocessing.format_converter import convert_to_wav
from src.preprocessing.noise_reducer import reduce_noise
from src.preprocessing.audio_segmenter import segment_by_silence
from src.recognition.speech_recognizer import SpeechRecognizer
from src.postprocessing.text_formatter import format_transcription
from src.postprocessing.timestamp_generator import add_timestamps
from src.output.file_writer import save_text, save_json, save_srt
from src.output.report_generator import generate_report

# Configure basic logging based on config
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(config.LOG_FILE),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger(__name__)

def process_audio(audio_source_type, source, language='auto', output_dir=config.OUTPUT_DIRECTORY, generate_rep=False):
    start_time = time.time()
    
    # --- PHASE 2: AUDIO INPUT ---
    try:
        if audio_source_type == 'file':
            logger.info(f"Loading files from {source}")
            audio_data = load_audio_file(source)
        elif audio_source_type == 'mic':
            # Here source parameter represents duration
            duration = int(source)
            logger.info("Recording from microphone...")
            numpy_audio = record_audio(duration, config.MICROPHONE_SAMPLE_RATE)
            # Create a temporary file to wrap into AudioSegment
            temp_mic_file = "./data/sample_audio/temp_mic.wav"
            save_recording(numpy_audio, temp_mic_file)
            audio_data = load_audio_file(temp_mic_file)
        else:
            raise ValueError(f"Unknown source type: {audio_source_type}")
            
        validation_report = validate_audio(audio_data)
        if not validation_report['is_valid']:
            logger.warning("Audio validation failed or returned warnings.")
            
    except Exception as e:
        logger.error(f"Input Error: {e}")
        return

    # --- PHASE 3: PREPROCESSING ---
    try:
        audio_data = convert_to_wav(audio_data, config.TARGET_SAMPLE_RATE)
        audio_data = reduce_noise(audio_data, config.NOISE_THRESHOLD)
        segments = segment_by_silence(audio_data, config.MIN_SILENCE_LENGTH, config.SILENCE_THRESHOLD)
    except Exception as e:
        logger.error(f"Preprocessing Error: {e}")
        return

    # --- PHASE 4: RECOGNITION ---
    try:
        recognizer = SpeechRecognizer(engine=config.DEFAULT_RECOGNIZER, language=language)
        transcription_results = recognizer.batch_transcribe(segments)
    except Exception as e:
        logger.error(f"Recognition Error: {e}")
        return

    # --- PHASE 5: POST-PROCESSING ---
    try:
        raw_text = transcription_results.get('transcription', '')
        formatted_text = format_transcription(raw_text)
        transcription_results['transcription'] = formatted_text
        
        # Adding timestamps to segments
        timestamps = add_timestamps(transcription_results.get('segments', []))
        srt_content = timestamps['srt']
    except Exception as e:
        logger.error(f"Post-processing Error: {e}")
        return

    # --- PHASE 6: OUTPUT ---
    try:
        processing_time = time.time() - start_time
        
        # Prepare Metadata
        base_name = "transcription_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if audio_source_type == 'file':
            base_name = os.path.splitext(os.path.basename(source))[0] + "_" + base_name
            
        out_base = os.path.join(output_dir, base_name)
        
        metadata = {
            "transcription": formatted_text,
            "language": transcription_results.get('language', language),
            "confidence": transcription_results.get('confidence', 0.0),
            "duration": validation_report['duration_seconds'],
            "timestamp_created": datetime.datetime.now().isoformat(),
            "segments": transcription_results.get('segments', []),
            
            # Additional for report
            "total_processing_time": processing_time,
            "audio_duration": validation_report['duration_seconds'],
            "segment_count": len(segments)
        }
        
        # Save Text
        if 'txt' in config.SAVE_FORMATS:
            save_text(formatted_text, out_base + ".txt")
            
        # Save JSON
        if 'json' in config.SAVE_FORMATS:
            save_json(metadata, out_base + ".json")
            
        # Save SRT
        if 'srt' in config.SAVE_FORMATS and config.INCLUDE_TIMESTAMPS:
            save_srt(srt_content, out_base + ".srt")
            
        # Print / Generating report
        if generate_rep:
            report = generate_report(metadata, out_base + "_report.txt")
            print(report)
        else:
            print("\n====================")
            print("Transcription Done.")
            print(f"Confidence: {(metadata['confidence']*100):.1f}%")
            print(f"Language: {metadata['language']}")
            print("====================\n")
            print(formatted_text[:1000] + ("..." if len(formatted_text) > 1000 else ""))
            
    except Exception as e:
        logger.error(f"Output Error: {e}")
        return
        
    logger.info("Pipeline finished successfully.")

def batch_process(input_dir, output_dir, language='auto'):
    if not os.path.isdir(input_dir):
        logger.error(f"Directory not found: {input_dir}")
        return
        
    for file in os.listdir(input_dir):
        if file.startswith('.'):
            continue
        filepath = os.path.join(input_dir, file)
        if os.path.isfile(filepath):
            logger.info(f"Processing batch file: {filepath}")
            process_audio('file', filepath, language, output_dir)

def main():
    parser = argparse.ArgumentParser(description="Multilingual Voice-to-Text System")
    parser.add_argument('--mode', type=str, choices=['mic', 'file', 'batch'], required=True, 
                        help='Input mode: mic, file, or batch processing')
    parser.add_argument('--duration', type=int, default=10, 
                        help='Duration in seconds for microphone recording')
    parser.add_argument('--input', type=str, 
                        help='Path to input audio file (for file mode)')
    parser.add_argument('--input-dir', type=str, 
                        help='Path to input directory (for batch mode)')
    parser.add_argument('--language', type=str, default=config.DEFAULT_LANGUAGE,
                        help='Language code (e.g., en-US) or auto for detection')
    parser.add_argument('--output', type=str, default=config.OUTPUT_DIRECTORY,
                        help='Path to output directory')
    parser.add_argument('--report', action='store_true', 
                        help='Generate full processing report')

    args = parser.parse_args()

    # Ensure output dir exists
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    os.makedirs('./data/sample_audio', exist_ok=True)

    if args.mode == 'mic':
        process_audio('mic', args.duration, args.language, args.output, args.report)
    elif args.mode == 'file':
        if not args.input:
            print("Error: --input is required for file mode")
            sys.exit(1)
        process_audio('file', args.input, args.language, args.output, args.report)
    elif args.mode == 'batch':
        if not args.input_dir:
            print("Error: --input-dir is required for batch mode")
            sys.exit(1)
        batch_process(args.input_dir, args.output, args.language)

if __name__ == "__main__":
    main()
