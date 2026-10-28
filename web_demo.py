"""Local web UI for the illegal-parking video pipeline.

The server intentionally uses only Python's standard library. Model inference is
run as a subprocess so the UI remains responsive while a video is processed.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from email.parser import BytesParser
from email.policy import default as email_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
INPUT_ROOT = ROOT / "input"
OUTPUT_ROOT = ROOT / "output"
DEFAULT_VIDEO = INPUT_ROOT / "demo_parking_video.mp4"
DEFAULT_CHECKPOINT = ROOT / "models" / "best.pth"
DEFAULT_ROI = "340,330;880,330;960,650;400,650"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
PROCESS_LINE = re.compile(r"Processed\s+(\d+)\s+frames\s+\(([\d.]+)s\)")


class JobStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.active_job: str | None = None

    def create(self, payload: dict) -> dict:
        with self.lock:
            if self.active_job:
                active = self.jobs.get(self.active_job, {})
                if active.get("status") in {"queued", "running", "encoding"}:
                    raise RuntimeError("Một video khác đang được xử lý")
            job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            job = {
                "id": job_id,
                "status": "queued",
                "message": "Đang chuẩn bị pipeline",
                "created_at": time.time(),
                "processed_frames": 0,
                "processed_seconds": 0.0,
                "logs": [],
                **payload,
            }
            self.jobs[job_id] = job
            self.active_job = job_id
            return dict(job)

    def update(self, job_id: str, **changes) -> None:
        with self.lock:
            job = self.jobs[job_id]
            job.update(changes)

    def append_log(self, job_id: str, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        with self.lock:
            logs = self.jobs[job_id]["logs"]
            logs.append(line)
            del logs[:-250]

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            public = dict(job)
            public.pop("input_path", None)
            public.pop("raw_output", None)
            public.pop("final_output", None)
            return public


STORE = JobStore()


def relative_url(path: Path) -> str:
    return "/files/" + quote(str(path.resolve().relative_to(ROOT.resolve())))


def summarize_csv(path: Path) -> dict:
    track_ids: set[str] = set()
    violation_ids: set[str] = set()
    classes: dict[str, set[str]] = {}
    row_count = 0
    max_stopped = 0.0
    if path.exists():
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                row_count += 1
                track_id = row.get("track_id", "")
                class_name = row.get("class", "unknown")
                if track_id:
                    track_ids.add(track_id)
                    classes.setdefault(class_name, set()).add(track_id)
                if row.get("violation") == "1" and track_id:
                    violation_ids.add(track_id)
                try:
                    max_stopped = max(max_stopped, float(row.get("stopped_duration_s", 0)))
                except ValueError:
                    pass
    return {
        "tracked_vehicles": len(track_ids),
        "violating_vehicles": len(violation_ids),
        "logged_observations": row_count,
        "max_stopped_seconds": round(max_stopped, 2),
        "classes": {name: len(ids) for name, ids in sorted(classes.items())},
    }


def transcode_for_browser(source: Path, destination: Path, job_id: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        STORE.append_log(job_id, "Không tìm thấy ffmpeg; giữ codec video gốc")
        shutil.copy2(source, destination)
        return False
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode == 0:
        return True
    STORE.append_log(job_id, "Không thể chuyển H.264; giữ codec video gốc")
    STORE.append_log(job_id, completed.stderr.splitlines()[-1] if completed.stderr else "ffmpeg failed")
    shutil.copy2(source, destination)
    return False


def run_job(job_id: str, device: str) -> None:
    with STORE.lock:
        internal = dict(STORE.jobs[job_id])
    output_dir = OUTPUT_ROOT / "web" / job_id
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    raw_output = videos_dir / "result_raw.mp4"
    final_output = videos_dir / "result.mp4"
    command = [
        sys.executable,
        "-m",
        "src.pipeline",
        str(internal["input_path"]),
        "--output",
        str(raw_output),
        "--checkpoint",
        str(DEFAULT_CHECKPOINT),
        "--roi",
        internal["roi"],
        "--confidence",
        str(internal["confidence"]),
        "--motion-threshold",
        str(internal["motion_threshold"]),
        "--motion-confirm-seconds",
        "2",
        "--motion-history-seconds",
        "4",
        "--parking-seconds",
        str(internal["parking_seconds"]),
        "--lost-track-seconds",
        "3",
        "--track-activation-threshold",
        "0.25",
        "--track-buffer",
        "30",
        "--matching-threshold",
        "0.80",
        "--device",
        device,
    ]
    STORE.update(job_id, status="running", message="Đang nhận diện và theo dõi phương tiện")
    STORE.append_log(job_id, "$ " + " ".join(command))
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            STORE.append_log(job_id, line)
            match = PROCESS_LINE.search(line)
            if match:
                STORE.update(
                    job_id,
                    processed_frames=int(match.group(1)),
                    processed_seconds=float(match.group(2)),
                )
        return_code = process.wait()
        if return_code != 0:
            STORE.update(
                job_id,
                status="failed",
                message=f"Pipeline kết thúc với mã lỗi {return_code}",
            )
            return

        STORE.update(job_id, status="encoding", message="Đang tối ưu video cho trình duyệt")
        transcode_for_browser(raw_output, final_output, job_id)
        log_path = output_dir / "logs" / "result_raw.csv"
        evidence_dir = output_dir / "violations"
        evidence = sorted(evidence_dir.glob("*.jpg")) if evidence_dir.exists() else []
        summary = summarize_csv(log_path)
        STORE.update(
            job_id,
            status="complete",
            message="Đã xử lý xong video",
            result_video=relative_url(final_output),
            raw_video=relative_url(raw_output),
            csv_url=relative_url(log_path),
            evidence=[relative_url(path) for path in evidence],
            summary=summary,
            finished_at=time.time(),
        )
    except Exception as exc:  # keep the web process alive and surface the error
        STORE.append_log(job_id, f"{type(exc).__name__}: {exc}")
        STORE.update(job_id, status="failed", message=str(exc))


def parse_roi(value: str) -> str:
    points: list[tuple[int, int]] = []
    try:
        for raw in value.split(";"):
            x_text, y_text = raw.split(",")
            points.append((int(x_text.strip()), int(y_text.strip())))
    except (ValueError, TypeError) as exc:
        raise ValueError("ROI phải có dạng x,y;x,y;x,y") from exc
    if len(points) < 3:
        raise ValueError("ROI cần ít nhất ba điểm")
    if any(x < 0 or y < 0 for x, y in points):
        raise ValueError("Tọa độ ROI không được âm")
    return ";".join(f"{x},{y}" for x, y in points)


def parse_multipart(handler: BaseHTTPRequestHandler, length: int) -> tuple[dict[str, str], str, bytes | None]:
    """Parse one browser multipart request without deprecated ``cgi``."""
    content_type = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise ValueError("Yêu cầu tải lên không đúng định dạng multipart")
    body = handler.rfile.read(length)
    envelope = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
        + body
    )
    message = BytesParser(policy=email_policy).parsebytes(envelope)
    fields: dict[str, str] = {}
    upload_name = ""
    upload_data: bytes | None = None
    if not message.is_multipart():
        raise ValueError("Không đọc được nội dung multipart")
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if name == "video" and filename:
            upload_name = Path(filename).name
            upload_data = payload
        elif name:
            fields[name] = payload.decode("utf-8", errors="replace")
    return fields, upload_name, upload_data


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "ParkingDemo/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[web] {self.address_string()} - {fmt % args}", flush=True)

    def json_response(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self.send_static(WEB_ROOT / "index.html")
        if parsed.path.startswith("/static/"):
            relative = parsed.path.removeprefix("/static/")
            target = (WEB_ROOT / relative).resolve()
            if WEB_ROOT.resolve() not in target.parents:
                return self.send_error(HTTPStatus.FORBIDDEN)
            return self.send_static(target)
        if parsed.path == "/api/config":
            return self.json_response(
                {
                    "default_video": relative_url(DEFAULT_VIDEO) if DEFAULT_VIDEO.exists() else None,
                    "default_roi": DEFAULT_ROI,
                    "device": self.server.device,
                    "checkpoint_exists": DEFAULT_CHECKPOINT.exists(),
                }
            )
        if parsed.path == "/api/status":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            job = STORE.get(job_id)
            if not job:
                return self.json_response({"error": "Không tìm thấy job"}, 404)
            return self.json_response(job)
        if parsed.path.startswith("/files/"):
            relative = unquote(parsed.path.removeprefix("/files/"))
            target = (ROOT / relative).resolve()
            allowed = (INPUT_ROOT.resolve(), OUTPUT_ROOT.resolve())
            if not any(target == base or base in target.parents for base in allowed):
                return self.send_error(HTTPStatus.FORBIDDEN)
            return self.send_file_with_range(target)
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/run":
            return self.send_error(HTTPStatus.NOT_FOUND)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                raise ValueError("Yêu cầu tải lên phải nhỏ hơn 512 MB")
            form, upload_name, upload_data = parse_multipart(self, length)
            roi = parse_roi(form.get("roi", DEFAULT_ROI))
            confidence = float(form.get("confidence", "0.30"))
            parking_seconds = float(form.get("parking_seconds", "5"))
            motion_threshold = float(form.get("motion_threshold", "8"))
            if not 0.01 <= confidence <= 1:
                raise ValueError("Confidence phải nằm trong [0.01, 1]")
            if not 1 <= parking_seconds <= 3600:
                raise ValueError("Thời gian đỗ phải nằm trong [1, 3600] giây")
            if not 0.1 <= motion_threshold <= 500:
                raise ValueError("Ngưỡng chuyển động phải nằm trong [0.1, 500]")

            input_path = DEFAULT_VIDEO
            if upload_name and upload_data is not None:
                suffix = Path(upload_name).suffix.lower()
                if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}:
                    raise ValueError("Định dạng video không được hỗ trợ")
                upload_dir = INPUT_ROOT / "web_uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                input_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
                with input_path.open("wb") as destination:
                    destination.write(upload_data)
            if not input_path.exists():
                raise ValueError("Chưa có video đầu vào")

            job = STORE.create(
                {
                    "input_path": input_path,
                    "source_name": upload_name or input_path.name,
                    "roi": roi,
                    "confidence": confidence,
                    "parking_seconds": parking_seconds,
                    "motion_threshold": motion_threshold,
                }
            )
            thread = threading.Thread(
                target=run_job,
                args=(job["id"], self.server.device),
                daemon=True,
            )
            thread.start()
            self.json_response({"job_id": job["id"]}, 202)
        except RuntimeError as exc:
            self.json_response({"error": str(exc)}, 409)
        except (ValueError, OSError) as exc:
            self.json_response({"error": str(exc)}, 400)

    def send_static(self, path: Path) -> None:
        if not path.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def send_file_with_range(self, path: Path) -> None:
        if not path.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        size = path.stat().st_size
        start, end = 0, max(0, size - 1)
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = min(int(match.group(2)), end)
                if start > end or start >= size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as file:
            file.seek(start)
            remaining = length
            while remaining:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    server.device = args.device
    print(f"Parking demo: http://{args.host}:{args.port} (device={args.device})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
