"""No-parking polygon helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence


Point = tuple[float, float]
Box = Sequence[float]


def bottom_center(box: Box) -> Point:
    if len(box) != 4:
        raise ValueError("A bounding box must contain x1, y1, x2, y2")
    x1, _y1, x2, y2 = map(float, box)
    return ((x1 + x2) / 2.0, y2)


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting test; points on polygon edges count as inside."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi)
        if abs(cross) < 1e-9 and min(xi, xj) <= x <= max(xi, xj) and min(
            yi, yj
        ) <= y <= max(yi, yj):
            return True
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (
            yj - yi
        ) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def parse_roi(text: str) -> list[tuple[int, int]]:
    """Parse ``x,y;x,y;...`` from the command line."""
    try:
        points = [tuple(map(int, item.split(","))) for item in text.split(";")]
    except ValueError as exc:
        raise ValueError("ROI must have the form 'x,y;x,y;x,y'") from exc
    if len(points) < 3 or any(len(point) != 2 for point in points):
        raise ValueError("ROI must contain at least three x,y points")
    return points


def load_roi(path: str | Path) -> list[tuple[int, int]]:
    """Load a polygon from either ``[[x,y], ...]`` or ``{'points': ...}``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw: Iterable[Iterable[int]] = data.get("points", []) if isinstance(data, dict) else data
    points = [tuple(map(int, point)) for point in raw]
    if len(points) < 3 or any(len(point) != 2 for point in points):
        raise ValueError("ROI JSON must contain at least three [x, y] points")
    return points

