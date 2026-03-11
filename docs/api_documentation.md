# API Reference

This document highlights the core classes and their respective signatures. Refer directly to the code for complex parameter explanations.

## `src.audio_input`
- **`record_audio(duration: float, sample_rate: int = 16000) -> np.ndarray`**
  Records numpy format frames directly off the default PyAudio input instance.
- **`load_audio_file(filepath: str) -> pydub.AudioSegment`**
  Loads local Audio objects into a unified abstraction.
- **`validate_audio(audio_data: AudioSegment) -> dict`**

## `src.preprocessing`
- **`convert_to_wav(audio_data: AudioSegment, target_sample_rate: int) -> AudioSegment`**
- **`reduce_noise(audio_data: AudioSegment, noise_threshold: int) -> AudioSegment`**
- **`segment_by_silence(audio_data: AudioSegment) -> List[Dict]`**

## `src.recognition`
- **`SpeechRecognizer(engine='google', language='auto')`**
  - **`transcribe(audio_data: AudioSegment, language: str) -> Dict`**
  - **`batch_transcribe(audio_segments: List[Dict]) -> Dict`**
- **`detect_language(audio_data: AudioSegment) -> str`**
- **`WhisperEngine(model_size='base')`**

## `src.postprocessing`
- **`format_transcription(text: str) -> str`**
- **`add_timestamps(transcription_chunks: List[Dict]) -> Dict`**

## `src.output`
- **`save_text(text: str, filepath: str) -> str`**
- **`save_json(data: Dict, filepath: str) -> str`**
- **`save_srt(timestamped_text: str, filepath: str) -> str`**
- **`generate_report(transcription_data: Dict) -> str`**
