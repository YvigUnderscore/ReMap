from __future__ import annotations

import json
from pathlib import Path
from flask import Flask, Response, jsonify, request, send_file

from .capabilities import detect_capabilities_payload
from .job_service import JobService
from .probe_service import build_option_payload, load_ocio_spaces, probe_inputs
from .settings_store import SettingsStore
from .server_controller import ServerController


def create_internal_app(
    settings_store: SettingsStore,
    job_service: JobService,
    server_controller: ServerController,
) -> Flask:
    app = Flask(__name__)

    @app.before_request
    def _handle_preflight():
        if request.method == "OPTIONS":
            return app.make_default_options_response()
        return None

    @app.after_request
    def _cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Range"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
        response.headers["Access-Control-Expose-Headers"] = "Content-Length,Content-Range,Accept-Ranges"
        return response

    @app.route("/internal/v1/health", methods=["GET"])
    def internal_health():
        return jsonify({"status": "ok", "service": "ReMap Desktop Backend"})

    @app.route("/internal/v1/settings", methods=["GET", "PUT"])
    def internal_settings():
        if request.method == "GET":
            return jsonify(settings_store.get().to_dict())
        payload = request.get_json(silent=True) or {}
        settings = settings_store.update(payload)
        return jsonify(settings.to_dict())

    @app.route("/internal/v1/system/capabilities", methods=["GET"])
    def system_capabilities():
        return jsonify(detect_capabilities_payload())

    @app.route("/internal/v1/options", methods=["GET"])
    def options():
        ocio_path = request.args.get("ocioPath")
        payload = build_option_payload()
        if ocio_path:
            payload["ocio_spaces"] = load_ocio_spaces(ocio_path)
            payload["default_ocio_config"] = ocio_path
        return jsonify(payload)

    @app.route("/internal/v1/probe", methods=["POST"])
    def probe():
        payload = request.get_json(silent=True) or {}
        input_mode = payload.get("input_mode", "video")
        input_paths = payload.get("input_paths", [])
        target_fps = payload.get("fps_extract")
        return jsonify(probe_inputs(input_mode, input_paths, target_fps))

    @app.route("/internal/v1/jobs", methods=["GET", "POST"])
    def jobs():
        if request.method == "GET":
            return jsonify({"jobs": job_service.list_jobs()})
        payload = request.get_json(silent=True) or {}
        try:
            job = job_service.create_job(payload)
        except ValueError as exc:
            return jsonify({"error": "Invalid Request", "message": str(exc)}), 400
        return jsonify(job.to_dict()), 202

    @app.route("/internal/v1/jobs/queue", methods=["DELETE"])
    def clear_job_queue():
        return jsonify(job_service.clear_queued_jobs())

    @app.route("/internal/v1/jobs/<job_id>", methods=["GET"])
    def job_detail(job_id: str):
        try:
            return jsonify(job_service.get_job(job_id).to_dict())
        except KeyError:
            return jsonify({"error": "Not Found", "message": f"Unknown job '{job_id}'"}), 404

    @app.route("/internal/v1/jobs/<job_id>", methods=["DELETE"])
    def delete_job(job_id: str):
        try:
            return jsonify(job_service.delete_job(job_id))
        except KeyError:
            return jsonify({"error": "Not Found", "message": f"Unknown job '{job_id}'"}), 404

    @app.route("/internal/v1/jobs/<job_id>/artifacts", methods=["GET"])
    def job_artifacts(job_id: str):
        try:
            return jsonify(job_service.get_artifacts(job_id))
        except KeyError:
            return jsonify({"error": "Not Found", "message": f"Unknown job '{job_id}'"}), 404

    @app.route("/internal/v1/files", methods=["GET"])
    def local_file():
        path = Path(request.args.get("path", ""))
        if not path.exists() or not path.is_file():
            return jsonify({"error": "Not Found", "message": "Unknown file"}), 404
        return send_file(path, conditional=True)

    @app.route("/internal/v1/jobs/<job_id>/cancel", methods=["POST"])
    def cancel_job(job_id: str):
        try:
            return jsonify(job_service.cancel_job(job_id).to_dict())
        except KeyError:
            return jsonify({"error": "Not Found", "message": f"Unknown job '{job_id}'"}), 404

    @app.route("/internal/v1/jobs/<job_id>/pause", methods=["POST"])
    def pause_job(job_id: str):
        try:
            return jsonify(job_service.pause_job(job_id).to_dict())
        except KeyError:
            return jsonify({"error": "Not Found", "message": f"Unknown job '{job_id}'"}), 404

    @app.route("/internal/v1/jobs/<job_id>/resume", methods=["POST"])
    def resume_job(job_id: str):
        try:
            return jsonify(job_service.resume_job(job_id).to_dict())
        except KeyError:
            return jsonify({"error": "Not Found", "message": f"Unknown job '{job_id}'"}), 404

    @app.route("/internal/v1/jobs/<job_id>/logs/stream", methods=["GET"])
    def job_logs_stream(job_id: str):
        try:
            last_event_id = int(request.args.get("lastEventId", "0"))
        except ValueError:
            last_event_id = 0

        def generate():
            try:
                for event in job_service.stream_logs(job_id, last_event_id=last_event_id):
                    yield f"id: {event['id']}\n"
                    yield "event: log\n"
                    yield f"data: {json.dumps(event)}\n\n"
            except KeyError:
                yield "event: error\n"
                yield f"data: {json.dumps({'message': 'Unknown job'})}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    @app.route("/internal/v1/server", methods=["GET", "PUT", "POST"])
    def internal_server():
        if request.method == "GET":
            return jsonify(server_controller.get_state())
        if request.method == "PUT":
            payload = request.get_json(silent=True) or {}
            config = server_controller.update_config(payload)
            return jsonify({"config": config, "state": server_controller.get_state()})
        payload = request.get_json(silent=True) or {}
        action = str(payload.get("action", "")).strip().lower()
        if action == "start":
            state = server_controller.start()
        elif action == "stop":
            state = server_controller.stop()
        else:
            state = server_controller.get_state()
        return jsonify(state)

    return app
