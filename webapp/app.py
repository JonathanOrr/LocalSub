import dataclasses
import json
import subprocess
import tempfile
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file

from pipeline.orchestrate import PipelineConfig
from pipeline.vad_ten import detect_raw_speech_runs
from pipeline.whisper_engine import WHISPER_LANGUAGES, extract_audio
from webapp.runner import JOBS, start_job, submit_confirm

app = Flask(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".ts", ".mpg", ".mpeg"}


@app.route("/")
def index():
    languages = sorted(WHISPER_LANGUAGES.items(), key=lambda kv: kv[1])
    return render_template("index.html", defaults=PipelineConfig(), languages=languages)


@app.route("/api/browse")
def browse():
    """List a directory's subfolders and video files, for the web UI's file-picker modal.
    We can't use a plain <input type="file"> for this: browsers deliberately don't expose
    the real filesystem path of a picked file to JS (only its bytes), and since the server
    and browser are the same local machine here, uploading a multi-GB video over HTTP just
    to learn its own path would be pointless - browsing by path server-side is direct."""
    raw = request.args.get("path") or str(Path.home())
    path = Path(raw).expanduser()
    try:
        path = path.resolve()
        if not path.is_dir():
            path = path.parent
        entries = []
        for entry in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if entry.is_dir():
                entries.append({"name": entry.name, "path": str(entry), "is_dir": True})
            elif entry.suffix.lower() in VIDEO_EXTENSIONS:
                entries.append({"name": entry.name, "path": str(entry), "is_dir": False})
    except OSError as e:
        return jsonify({"error": str(e)}), 400
    parent = str(path.parent) if path.parent != path else None
    return jsonify({"path": str(path), "parent": parent, "entries": entries})


@app.route("/api/vad_analyze", methods=["POST"])
def vad_analyze():
    """Extract audio from a real video and run TEN VAD's frame classification (only - no
    merge/discard/pad/split yet) so the web UI's VAD diagram can preview real detected
    speech instead of its synthetic example. Only re-run when the threshold changes - every
    other VAD knob is cheap post-processing math the frontend already redoes instantly in
    JS (computeVadSegments) against whatever raw_runs this returns."""
    data = request.get_json(force=True)
    video_path = Path(data.get("video_path", ""))
    try:
        threshold = float(data.get("vad_threshold", 0.5))
    except (TypeError, ValueError):
        return jsonify({"error": "vad_threshold must be a number"}), 400
    if not video_path.exists():
        return jsonify({"error": f"video not found: {video_path}"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            wav_path = extract_audio(video_path, Path(tmpdir))
            raw_runs, duration_s = detect_raw_speech_runs(wav_path, threshold)
        except SystemExit as e:
            return jsonify({"error": str(e)}), 400
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"ffmpeg failed: {e}"}), 500

    return jsonify({"raw_runs": raw_runs, "duration_s": duration_s})


@app.route("/api/audio_clip")
def audio_clip():
    """Extract a short audio clip on demand (not pre-extracted/cached - a single short clip
    is fast enough that there's no real benefit to caching, and it avoids needing to think
    about cache invalidation) so the web UI can let you click a VAD segment and hear it."""
    raw_path = request.args.get("path", "")
    start_raw, end_raw = request.args.get("start"), request.args.get("end")
    if start_raw is None or end_raw is None:
        return jsonify({"error": "start/end are required"}), 400
    try:
        start, end = float(start_raw), float(end_raw)
    except ValueError:
        return jsonify({"error": "start/end must be numbers"}), 400
    if start < 0 or end <= start:
        return jsonify({"error": "invalid start/end range"}), 400

    video_path = Path(raw_path)
    if not video_path.exists():
        return jsonify({"error": f"video not found: {video_path}"}), 400

    cmd = [
        "ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(video_path),
        "-t", f"{end - start:.3f}", "-vn", "-f", "mp3", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        return jsonify({"error": proc.stderr.decode(errors="replace")}), 500
    return Response(proc.stdout, mimetype="audio/mpeg")


@app.route("/api/video")
def video_file():
    """Serve the video file itself, Range-request aware (Flask's conditional send_file
    handles the 206 Partial Content dance), so the reference-frame picker can use a real
    <video> element with native seeking instead of guessing timestamps blind. Not all
    accepted source formats play back in a browser (e.g. .avi/.wmv rarely do) - the manual
    timestamp field stays as a fallback for those."""
    raw_path = request.args.get("path", "")
    video_path = Path(raw_path)
    if not video_path.exists():
        return jsonify({"error": f"video not found: {video_path}"}), 400
    return send_file(video_path, conditional=True)


@app.route("/api/frame_preview")
def frame_preview():
    """Extract a single frame at a given timestamp, on demand - powers the reference-frame
    picker in the web UI, so you can scrub to a moment and see it before pinning it as a
    labeled reference (see PipelineConfig.reference_frames)."""
    raw_path = request.args.get("path", "")
    t_raw = request.args.get("t")
    if t_raw is None:
        return jsonify({"error": "t is required"}), 400
    try:
        t = float(t_raw)
    except ValueError:
        return jsonify({"error": "t must be a number"}), 400
    if t < 0:
        return jsonify({"error": "t must be >= 0"}), 400

    video_path = Path(raw_path)
    if not video_path.exists():
        return jsonify({"error": f"video not found: {video_path}"}), 400

    cmd = [
        "ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video_path),
        "-frames:v", "1", "-q:v", "3", "-f", "image2", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return jsonify({"error": proc.stderr.decode(errors="replace") or "no frame extracted"}), 500
    return Response(proc.stdout, mimetype="image/jpeg")


@app.route("/api/jobs", methods=["POST"])
def create_job():
    data = request.get_json(force=True)
    video_path = Path(data["video_path"])

    config_kwargs = {}
    for f in dataclasses.fields(PipelineConfig):
        if f.name not in data:
            continue
        value = data[f.name]
        if f.name == "workdir":
            value = Path(value) if value else None
        config_kwargs[f.name] = value
    config = PipelineConfig(**config_kwargs)

    job_id = start_job(video_path, config)
    return jsonify({"job_id": job_id})


@app.route("/api/jobs/<job_id>/events")
def job_events(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404

    def stream():
        while True:
            item = job.events.get()
            yield f"data: {json.dumps(item)}\n\n"
            if item.get("type") in ("done", "error"):
                break

    return Response(stream(), mimetype="text/event-stream")


@app.route("/api/jobs/<job_id>/confirm", methods=["POST"])
def job_confirm(job_id):
    data = request.get_json(force=True)
    ok = submit_confirm(job_id, data)
    return jsonify({"ok": ok})


@app.route("/api/jobs/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({"status": job.status, "result": job.result, "error": job.error})
