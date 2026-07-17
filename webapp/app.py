import dataclasses
import json
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from pipeline.orchestrate import PipelineConfig
from pipeline.whisper_engine import WHISPER_LANGUAGES
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
