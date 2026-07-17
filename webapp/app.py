import dataclasses
import json
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from pipeline.orchestrate import PipelineConfig
from webapp.runner import JOBS, start_job, submit_confirm

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", defaults=PipelineConfig())


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
