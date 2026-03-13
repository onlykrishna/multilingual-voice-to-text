"""
Flask web server for the Multilingual Voice-to-Text System.
"""
import os, sys, json, time, datetime, threading, glob, logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, stream_with_context
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))

import config
from src.audio_input.file_loader import load_audio_file, validate_audio
from src.preprocessing.format_converter import convert_to_wav
from src.preprocessing.noise_reducer import reduce_noise
from src.preprocessing.audio_segmenter import segment_by_silence
from src.recognition.speech_recognizer import SpeechRecognizer
from src.postprocessing.text_formatter import format_transcription
from src.postprocessing.timestamp_generator import add_timestamps
from src.output.file_writer import save_text, save_json, save_srt
from src.output.report_generator import generate_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB

UPLOAD_FOLDER = './data/uploads'
OUTPUT_FOLDER = './data/output'
for d in [UPLOAD_FOLDER, OUTPUT_FOLDER, './logs']:
    os.makedirs(d, exist_ok=True)

# ── SSE event store ──────────────────────────────────────────────────────────
_lock = threading.Lock()
_sse_events: dict[str, list] = {}

def push_event(sid, data):
    with _lock:
        if sid not in _sse_events:
            _sse_events[sid] = []
        _sse_events[sid].append(json.dumps(data))

# ── Pipeline ─────────────────────────────────────────────────────────────────
def run_pipeline(source_type, source, language, engine, session_id, formats):
    start = time.time()

    def step(msg, pct, **extra):
        push_event(session_id, {"message": msg, "progress": pct, **extra})

    try:
        # 1. Load / record
        if source_type == "mic":
            duration = int(source)
            step(f"🎙️ Recording {duration}s from microphone…", 5, step="record")
            from src.audio_input.microphone_capture import record_audio, save_recording
            np_audio = record_audio(duration, config.MICROPHONE_SAMPLE_RATE)
            tmp_path = os.path.join(UPLOAD_FOLDER, f"mic_{session_id}.wav")
            save_recording(np_audio, tmp_path)
            audio = load_audio_file(tmp_path)
        else:
            step("📂 Loading audio file…", 8, step="load")
            audio = load_audio_file(source)

        v = validate_audio(audio)
        step(f"✅ Loaded: {v['duration_seconds']:.1f}s · {v['sample_rate']}Hz · "
             f"{'stereo' if v['channels']==2 else 'mono'}", 18, step="validated", meta=v)

        # 2. Preprocess
        step("🔄 Converting to 16kHz mono…", 28, step="convert")
        audio = convert_to_wav(audio, config.TARGET_SAMPLE_RATE)

        step("🔇 Noise reduction (band-pass 300–3400 Hz)…", 38, step="denoise")
        audio = reduce_noise(audio, config.NOISE_THRESHOLD)

        step("✂️ Segmenting by silence…", 48, step="segment")
        segments = segment_by_silence(audio, config.MIN_SILENCE_LENGTH, config.SILENCE_THRESHOLD)
        step(f"📊 {len(segments)} segment(s) found", 54, step="segmented")

        # 3. Transcribe
        step(f"🤖 Transcribing with {engine.upper()} engine…", 60, step="transcribe")
        rec = SpeechRecognizer(engine=engine, language=language)
        result = rec.batch_transcribe(segments)

        # 4. Post-process
        step("📝 Formatting transcription…", 78, step="format")
        formatted = format_transcription(result.get("transcription", ""))
        result["transcription"] = formatted

        step("⏱️ Generating SRT/VTT timestamps…", 86, step="timestamp")
        ts = add_timestamps(result.get("segments", []))

        # 5. Save outputs
        step("💾 Saving output files…", 92, step="save")
        elapsed  = time.time() - start
        base     = f"transcription_{session_id}"
        out_base = os.path.join(OUTPUT_FOLDER, base)

        metadata = {
            "transcription": formatted,
            "language": result.get("language", language),
            "confidence": result.get("confidence", 0.0),
            "duration": v["duration_seconds"],
            "timestamp_created": datetime.datetime.now().isoformat(),
            "segments": result.get("segments", []),
            "total_processing_time": elapsed,
            "audio_duration": v["duration_seconds"],
            "segment_count": len(segments),
        }

        saved = {}
        if "txt"  in formats: saved["txt"]  = os.path.basename(save_text(formatted, out_base + ".txt"))
        if "json" in formats: saved["json"] = os.path.basename(save_json(metadata,  out_base + ".json"))
        if "srt"  in formats and ts.get("srt"):
            saved["srt"] = os.path.basename(save_srt(ts["srt"], out_base + ".srt"))

        report = generate_report(metadata)

        step("🎉 Done!", 100, step="done",
             transcription=formatted,
             language=result.get("language", language),
             confidence=round(result.get("confidence", 0.0) * 100, 1),
             duration=round(v["duration_seconds"], 2),
             processing_time=round(elapsed, 2),
             segment_count=len(segments),
             word_count=len(formatted.split()) if formatted else 0,
             srt=ts.get("srt", ""),
             vtt=ts.get("vtt", ""),
             saved_files=saved,
             report=report)

    except Exception as e:
        logger.error("Pipeline failed", exc_info=True)
        push_event(session_id, {"step": "error", "message": f"❌ {e}", "progress": 0})

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools():
    return jsonify({}), 200

@app.route("/api/mic", methods=["POST"])
def api_mic():
    d = request.json or {}
    sid      = d.get("session_id", str(int(time.time())))
    duration = int(d.get("duration", 10))
    language = d.get("language", "auto")
    engine   = d.get("engine", "google")
    formats  = d.get("formats", ["txt", "json", "srt"])
    with _lock:
        _sse_events[sid] = []
    threading.Thread(target=run_pipeline,
                     args=("mic", duration, language, engine, sid, formats),
                     daemon=True).start()
    return jsonify({"session_id": sid, "status": "started"})

@app.route("/api/file", methods=["POST"])
def api_file():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    f       = request.files["audio"]
    sid     = request.form.get("session_id", str(int(time.time())))
    lang    = request.form.get("language", "auto")
    engine  = request.form.get("engine", "google")
    formats = request.form.get("formats", "txt,json,srt").split(",")

    ext = Path(f.filename).suffix.lower()
    if ext not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    path = os.path.join(UPLOAD_FOLDER, f"upload_{sid}{ext}")
    f.save(path)
    with _lock:
        _sse_events[sid] = []
    threading.Thread(target=run_pipeline,
                     args=("file", path, lang, engine, sid, formats),
                     daemon=True).start()
    return jsonify({"session_id": sid, "status": "started"})

@app.route("/api/events/<sid>")
def sse_stream(sid):
    """Server-Sent Events — streams pipeline progress to the browser."""
    def generate():
        seen    = 0
        timeout = 180
        t0      = time.time()
        while time.time() - t0 < timeout:
            with _lock:
                evts = list(_sse_events.get(sid, []))
            while seen < len(evts):
                yield f"data: {evts[seen]}\n\n"
                seen += 1
            if seen:
                last = json.loads(evts[seen - 1])
                if last.get("step") in ("done", "error"):
                    break
            time.sleep(0.25)
        with _lock:
            _sse_events.pop(sid, None)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.route("/api/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

@app.route("/api/outputs")
def list_outputs():
    files = []
    for f in sorted(glob.glob(os.path.join(OUTPUT_FOLDER, "*")), reverse=True)[:30]:
        s = os.stat(f)
        files.append({
            "name": os.path.basename(f),
            "size": s.st_size,
            "modified": datetime.datetime.fromtimestamp(s.st_mtime).strftime("%Y-%m-%d %H:%M")
        })
    return jsonify(files)

# ══════════════════════════════════════════════════════════════════════
#  VOICE IDENTIFICATION API
# ══════════════════════════════════════════════════════════════════════

@app.route("/api/enroll-voice", methods=["POST"])
def api_enroll_voice():
    """Upload a sample voice recording and create a speaker profile."""
    if "sample" not in request.files:
        return jsonify({"success": False, "error": "No sample file uploaded"}), 400

    f   = request.files["sample"]
    ext = Path(f.filename).suffix.lower()
    if ext not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}:
        return jsonify({"success": False, "error": f"Unsupported format: {ext}"}), 400

    name = request.form.get("speaker_name", "").strip() or None

    # Save upload
    tmp = os.path.join(UPLOAD_FOLDER, f"enroll_{int(time.time())}{ext}")
    f.save(tmp)

    try:
        from src.voice_identification.enrollment import VoiceEnrollment
        enroller  = VoiceEnrollment(profiles_dir=config.VOICE_PROFILES_DIR)
        validation = enroller.validate_sample(tmp)

        if not validation["is_valid"]:
            return jsonify({
                "success": False,
                "error":   "Sample quality too low",
                "details": validation,
            }), 422

        profile = enroller.create_profile(tmp, speaker_name=name)
        return jsonify({
            "success":      True,
            "speaker_id":   profile["speaker_id"],
            "name":         profile["metadata"]["name"],
            "quality_score": profile["metadata"]["quality_score"],
            "snr":          profile["metadata"]["snr"],
            "duration":     profile["metadata"]["sample_duration"],
            "issues":       validation["issues"],
        })
    except Exception as e:
        logger.error("Enrollment failed", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _run_voice_id_pipeline(speaker_id: str, conv_path: str, session_id: str):
    """Background thread: diarize → match → transcribe → report → SSE."""
    from src.voice_identification.enrollment       import VoiceEnrollment
    from src.voice_identification.speaker_diarizer import SpeakerDiarizer
    from src.voice_identification.voice_matcher    import VoiceMatcher
    from src.voice_identification.segment_extractor import SegmentExtractor
    from src.voice_identification.analytics_generator import AnalyticsGenerator
    from src.recognition.speech_recognizer        import SpeechRecognizer
    from pydub import AudioSegment as _AS

    def step(msg, pct, **extra):
        push_event(session_id, {"message": msg, "progress": pct, **extra})

    try:
        # 1. Load enrolled profile
        step("👤 Loading speaker profile…", 5, step="load_profile")
        enroller = VoiceEnrollment(profiles_dir=config.VOICE_PROFILES_DIR)
        profile  = enroller.load_profile(speaker_id)
        ref_emb  = profile["embedding"]

        # 2. Diarize conversation
        step("🔍 Segmenting conversation by speaker…", 15, step="diarize")
        diarizer = SpeakerDiarizer(
            window_s    = config.DIARIZER_WINDOW_S,
            hop_s       = config.DIARIZER_HOP_S,
            min_speakers= config.MIN_SPEAKERS,
            max_speakers= config.MAX_SPEAKERS,
            hf_token    = config.HF_TOKEN,
        )
        all_segs = diarizer.diarize(conv_path)
        wav_arr  = diarizer.get_wav(conv_path)
        step(f"📊 {len(all_segs)} speaker segment(s) detected", 35, step="diarized")

        # 3. Match against reference
        step("🎯 Comparing voice embeddings…", 45, step="match")
        matcher  = VoiceMatcher(similarity_threshold=config.SIMILARITY_THRESHOLD)
        matched_all = matcher.match_segments(ref_emb, all_segs, wav_arr)
        target_segs = [s for s in matched_all if s["is_match"]]
        step(f"✅ {len(target_segs)} segment(s) matched target speaker", 60, step="matched")

        # 4. Transcribe matched segments
        step("🤖 Transcribing target speaker's dialogue (Whisper)…", 65, step="transcribe")
        texts = {}
        if target_segs:
            audio_full = _AS.from_file(conv_path)
            rec = SpeechRecognizer(engine="whisper", language="auto")
            for i, seg in enumerate(target_segs):
                chunk = audio_full[int(seg["start"]*1000): int(seg["end"]*1000)]
                from src.preprocessing.format_converter import convert_to_wav as _cvt
                chunk = _cvt(chunk, 16000)
                r     = rec.transcribe(chunk)
                texts[i] = r.get("text", "")
                pct = 65 + int(25 * (i + 1) / len(target_segs))
                step(f"  Transcribed segment {i+1}/{len(target_segs)}", pct, step="transcribe_seg")

        # 5. Extract audio
        step("🎵 Extracting target speaker audio…", 88, step="extract")
        extractor  = SegmentExtractor()
        target_audio = extractor.extract_segments(conv_path, target_segs, config.SEGMENT_PADDING_MS)
        target_wav_path = os.path.join(OUTPUT_FOLDER, f"target_speaker_{session_id}.wav")
        extractor.save_extracted_audio(target_audio, target_wav_path)

        # 6. Generate report + SVG
        step("📊 Generating analytics report…", 93, step="report")
        conv_duration = len(wav_arr) / 16_000
        analytics = AnalyticsGenerator()
        report    = analytics.generate_report(
            enrollment_data      = profile,
            conversation_duration= conv_duration,
            all_segments         = matched_all,
            matched_segments     = target_segs,
            transcriptions       = texts,
        )
        svg = analytics.generate_svg_timeline(report["timeline"], conv_duration)

        # Save JSON report
        report_path = os.path.join(OUTPUT_FOLDER, f"voice_id_report_{session_id}.json")
        analytics.save_report(report, report_path)

        step("🎉 Voice Identification Complete!", 100, step="done",
             report            = report,
             svg_timeline      = svg,
             target_audio_file = os.path.basename(target_wav_path),
             report_file       = os.path.basename(report_path))

    except Exception as e:
        logger.error("Voice ID pipeline failed", exc_info=True)
        push_event(session_id, {"step": "error", "message": f"❌ {e}", "progress": 0})
    finally:
        if os.path.exists(conv_path):
            os.remove(conv_path)


@app.route("/api/detect-speaker", methods=["POST"])
def api_detect_speaker():
    """Upload a conversation and find the enrolled speaker."""
    if "conversation" not in request.files:
        return jsonify({"success": False, "error": "No conversation file"}), 400

    f          = request.files["conversation"]
    speaker_id = request.form.get("speaker_id", "").strip()
    if not speaker_id:
        return jsonify({"success": False, "error": "speaker_id is required"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}:
        return jsonify({"success": False, "error": f"Unsupported format: {ext}"}), 400

    sid  = f"vid_{int(time.time())}"
    path = os.path.join(UPLOAD_FOLDER, f"conv_{sid}{ext}")
    f.save(path)

    with _lock:
        _sse_events[sid] = []

    threading.Thread(
        target=_run_voice_id_pipeline,
        args=(speaker_id, path, sid),
        daemon=True,
    ).start()

    return jsonify({"success": True, "session_id": sid})


@app.route("/api/speaker-profiles", methods=["GET"])
def api_list_profiles():
    from src.voice_identification.enrollment import VoiceEnrollment
    return jsonify(VoiceEnrollment(profiles_dir=config.VOICE_PROFILES_DIR).list_profiles())


@app.route("/api/speaker-profiles/<speaker_id>", methods=["DELETE"])
def api_delete_profile(speaker_id):
    from src.voice_identification.enrollment import VoiceEnrollment
    deleted = VoiceEnrollment(profiles_dir=config.VOICE_PROFILES_DIR).delete_profile(speaker_id)
    return jsonify({"deleted": deleted})


@app.route("/api/download-target-audio/<session_id>")
def api_download_target_audio(session_id):
    fname = f"target_speaker_{session_id}.wav"
    return send_from_directory(OUTPUT_FOLDER, fname, as_attachment=True)



if __name__ == "__main__":
    print("\n" + "═" * 52)
    print("  🎙️  VoiceScribe — Web Interface")
    print("  → http://localhost:5050")
    print("═" * 52 + "\n")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)

