"""Thin adapter around Supervision's ByteTrack implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Track:
    track_id: int
    box: np.ndarray
    confidence: float
    class_id: int


class ByteTrackAdapter:
    def __init__(
        self,
        *,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ) -> None:
        try:
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError(
                "ByteTrack requires 'supervision'. Run: pip install -r requirements.txt"
            ) from exc
        self._sv = sv
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> list[Track]:
        detections = self._sv.Detections(
            xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            confidence=np.asarray(scores, dtype=np.float32),
            class_id=np.asarray(labels, dtype=int),
        )
        tracked = self._tracker.update_with_detections(detections)
        results: list[Track] = []
        if tracked.tracker_id is None:
            return results
        for box, confidence, class_id, track_id in zip(
            tracked.xyxy,
            tracked.confidence,
            tracked.class_id,
            tracked.tracker_id,
        ):
            if track_id is None:
                continue
            results.append(
                Track(
                    int(track_id),
                    np.asarray(box, dtype=np.float32),
                    float(confidence),
                    int(class_id),
                )
            )
        return results

