"""Time-based, jitter-resistant motion analysis."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True, slots=True)
class MotionResult:
    is_stopped: bool
    displacement_px: float
    observed_seconds: float


class MotionAnalyzer:
    """Classify tracks by their spread over a recent time window.

    Using the maximum distance from the first point is more resistant to tiny
    alternating detector jitter than adding frame-to-frame displacements.
    """

    def __init__(
        self,
        threshold_px: float = 8.0,
        confirmation_seconds: float = 2.0,
        history_seconds: float = 4.0,
    ) -> None:
        if threshold_px < 0 or confirmation_seconds <= 0:
            raise ValueError("Motion thresholds must be positive")
        if history_seconds < confirmation_seconds:
            raise ValueError("history_seconds must be >= confirmation_seconds")
        self.threshold_px = threshold_px
        self.confirmation_seconds = confirmation_seconds
        self.history_seconds = history_seconds
        self._history: dict[int, deque[tuple[float, tuple[float, float]]]] = defaultdict(
            deque
        )

    def update(
        self, track_id: int, point: tuple[float, float], timestamp: float
    ) -> MotionResult:
        history = self._history[track_id]
        # Protect the state machine against a broken/non-monotonic video clock.
        if history and timestamp < history[-1][0]:
            history.clear()
        history.append((timestamp, point))
        cutoff = timestamp - self.history_seconds
        # Retain one sample immediately before the history boundary.  The same
        # rule below prevents discrete frame timestamps from producing a window
        # that is forever a fraction shorter than confirmation_seconds.
        while len(history) > 1 and history[1][0] < cutoff:
            history.popleft()

        analysis_cutoff = timestamp - self.confirmation_seconds
        items = list(history)
        anchor_index = 0
        for index, (sample_time, _point) in enumerate(items):
            if sample_time <= analysis_cutoff:
                anchor_index = index
            else:
                break
        recent = items[anchor_index:]
        observed = timestamp - recent[0][0]
        origin = recent[0][1]
        displacement = max(
            hypot(candidate[0] - origin[0], candidate[1] - origin[1])
            for _, candidate in recent
        )
        stopped = (
            observed >= self.confirmation_seconds
            and displacement <= self.threshold_px
        )
        return MotionResult(stopped, displacement, observed)

    def remove(self, track_id: int) -> None:
        self._history.pop(track_id, None)
