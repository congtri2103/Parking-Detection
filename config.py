"""Default configuration for the illegal-parking pipeline.

Values here are deliberately conservative starting points.  Camera geometry and
the motion threshold must be calibrated for each installation.
"""

from dataclasses import dataclass, field
from pathlib import Path


CLASS_NAMES = {1: "car", 2: "motorcycle"}
NUM_CLASSES = 3  # background + car + motorcycle


@dataclass(slots=True)
class DetectorConfig:
    checkpoint: Path | None = Path("models/best.pth")
    confidence_threshold: float = 0.50
    nms_threshold: float = 0.50
    min_size: int = 800
    max_size: int = 1333


@dataclass(slots=True)
class MotionConfig:
    # Pixel radius in which the bottom-centre must remain during the window.
    threshold_px: float = 8.0
    # A track needs this much history before it may be called stationary.
    confirmation_seconds: float = 2.0
    history_seconds: float = 4.0


@dataclass(slots=True)
class ParkingConfig:
    parking_seconds: float = 10.0
    lost_track_seconds: float = 3.0


@dataclass(slots=True)
class TrackerConfig:
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.80
    frame_rate: int = 30


@dataclass(slots=True)
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    parking: ParkingConfig = field(default_factory=ParkingConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    # Empty by default on purpose: every fixed camera needs its own polygon.
    no_parking_roi: list[tuple[int, int]] = field(default_factory=list)

