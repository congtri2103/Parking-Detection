"""Illegal-parking state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ParkingStatus(str, Enum):
    MOVING = "MOVING"
    STOPPED = "STOPPED"
    PARKED = "PARKED"


@dataclass(frozen=True, slots=True)
class ParkingResult:
    status: ParkingStatus
    stopped_duration: float
    violation: bool
    violation_started: bool


@dataclass(slots=True)
class _TrackState:
    stopped_since: float | None = None
    violating: bool = False
    last_seen: float = 0.0


class ParkingStateMachine:
    def __init__(self, parking_seconds: float = 10.0) -> None:
        if parking_seconds < 0:
            raise ValueError("parking_seconds cannot be negative")
        self.parking_seconds = parking_seconds
        self._states: dict[int, _TrackState] = {}

    def update(
        self,
        track_id: int,
        *,
        is_stopped: bool,
        inside_roi: bool,
        timestamp: float,
    ) -> ParkingResult:
        state = self._states.setdefault(track_id, _TrackState(last_seen=timestamp))
        state.last_seen = timestamp

        # Only time stationary presence inside the prohibited zone. Leaving the
        # zone or moving starts a new parking episode.
        if not is_stopped:
            state.stopped_since = None
            state.violating = False
            return ParkingResult(ParkingStatus.MOVING, 0.0, False, False)

        if not inside_roi:
            state.stopped_since = None
            state.violating = False
            return ParkingResult(ParkingStatus.STOPPED, 0.0, False, False)

        if state.stopped_since is None:
            state.stopped_since = timestamp
        duration = max(0.0, timestamp - state.stopped_since)
        violation = duration >= self.parking_seconds
        started = violation and not state.violating
        state.violating = violation
        status = ParkingStatus.PARKED if violation else ParkingStatus.STOPPED
        return ParkingResult(status, duration, violation, started)

    def expire(self, current_time: float, max_age: float) -> list[int]:
        expired = [
            track_id
            for track_id, state in self._states.items()
            if current_time - state.last_seen > max_age
        ]
        for track_id in expired:
            del self._states[track_id]
        return expired
