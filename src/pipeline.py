"""End-to-end illegal-parking detection for a fixed-camera video."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import CLASS_NAMES
from models.detector import load_model
from src.inference import detect_frame, select_device
from src.motion import MotionAnalyzer
from src.parking_state import ParkingStateMachine, ParkingStatus
from src.roi import bottom_center, load_roi, parse_roi, point_in_polygon
from src.tracker import ByteTrackAdapter, Track


CSV_FIELDS = [
    "video_time_s",
    "frame",
    "track_id",
    "class",
    "confidence",
    "zone",
    "motion_px",
    "stopped_duration_s",
    "state",
    "violation",
]


def draw_track(frame, track: Track, result, inside: bool) -> None:
    x1, y1, x2, y2 = map(int, track.box)
    color = (0, 0, 255) if result.violation else ((0, 180, 255) if inside else (0, 210, 0))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    lines = [
        f"ID {track.track_id} | {CLASS_NAMES.get(track.class_id, str(track.class_id))} {track.confidence:.2f}",
        f"{result.status.value} {result.stopped_duration:.1f}s | {'NO PARKING' if inside else 'OUTSIDE'}",
    ]
    if result.violation:
        lines.append("VIOLATION")
    for index, line in enumerate(lines):
        y = max(18, y1 - 8 - (len(lines) - 1 - index) * 20)
        cv2.putText(frame, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input video path")
    parser.add_argument("--output", default="output/videos/result.mp4")
    parser.add_argument("--checkpoint", default="models/best.pth")
    roi_group = parser.add_mutually_exclusive_group(required=True)
    roi_group.add_argument("--roi", help="Polygon as 'x,y;x,y;x,y;...' ")
    roi_group.add_argument("--roi-file", help="JSON polygon file")
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--motion-threshold", type=float, default=8.0)
    parser.add_argument("--motion-confirm-seconds", type=float, default=2.0)
    parser.add_argument("--motion-history-seconds", type=float, default=4.0)
    parser.add_argument("--parking-seconds", type=float, default=10.0)
    parser.add_argument("--lost-track-seconds", type=float, default=3.0)
    parser.add_argument("--track-activation-threshold", type=float, default=0.25)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--matching-threshold", type=float, default=0.80)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roi = parse_roi(args.roi) if args.roi else load_roi(args.roi_file)
    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Input video has invalid dimensions")

    output = Path(args.output)
    artifact_root = output.parent.parent if output.parent.name == "videos" else output.parent
    evidence_dir = artifact_root / "violations"
    log_dir = artifact_root / "logs"
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output}")

    device = select_device(args.device)
    model = load_model(args.checkpoint, device)
    tracker = ByteTrackAdapter(
        track_activation_threshold=args.track_activation_threshold,
        lost_track_buffer=args.track_buffer,
        minimum_matching_threshold=args.matching_threshold,
        frame_rate=max(1, round(fps)),
    )
    motion = MotionAnalyzer(
        args.motion_threshold,
        args.motion_confirm_seconds,
        args.motion_history_seconds,
    )
    parking = ParkingStateMachine(args.parking_seconds)
    polygon = np.asarray(roi, dtype=np.int32)
    log_path = log_dir / f"{output.stem}.csv"
    frame_index = 0

    try:
        with log_path.open("w", newline="", encoding="utf-8") as log_file:
            log = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
            log.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_index / fps
                boxes, scores, labels = detect_frame(model, frame, device, args.confidence)
                supported = np.isin(labels, list(CLASS_NAMES))
                tracks = tracker.update(boxes[supported], scores[supported], labels[supported])

                cv2.polylines(frame, [polygon], True, (0, 165, 255), 2)
                cv2.putText(frame, "NO PARKING", tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
                for track in tracks:
                    point = bottom_center(track.box)
                    inside = point_in_polygon(point, roi)
                    motion_result = motion.update(track.track_id, point, timestamp)
                    result = parking.update(
                        track.track_id,
                        is_stopped=motion_result.is_stopped,
                        inside_roi=inside,
                        timestamp=timestamp,
                    )
                    draw_track(frame, track, result, inside)
                    cv2.circle(frame, tuple(map(int, point)), 4, (255, 255, 255), -1)
                    log.writerow(
                        {
                            "video_time_s": f"{timestamp:.3f}",
                            "frame": frame_index,
                            "track_id": track.track_id,
                            "class": CLASS_NAMES.get(track.class_id, track.class_id),
                            "confidence": f"{track.confidence:.4f}",
                            "zone": "NO_PARKING" if inside else "OUTSIDE",
                            "motion_px": f"{motion_result.displacement_px:.2f}",
                            "stopped_duration_s": f"{result.stopped_duration:.3f}",
                            "state": result.status.value,
                            "violation": int(result.violation),
                        }
                    )
                    if result.violation_started:
                        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        evidence = evidence_dir / f"violation_ID{track.track_id}_{stamp}.jpg"
                        cv2.imwrite(str(evidence), frame)

                for expired_id in parking.expire(timestamp, args.lost_track_seconds):
                    motion.remove(expired_id)
                writer.write(frame)
                frame_index += 1
                if frame_index % max(1, round(fps * 5)) == 0:
                    print(f"Processed {frame_index} frames ({timestamp:.1f}s)", flush=True)
    finally:
        capture.release()
        writer.release()
    print(f"Video: {output}\nLog: {log_path}\nEvidence: {evidence_dir}")


if __name__ == "__main__":
    main()
