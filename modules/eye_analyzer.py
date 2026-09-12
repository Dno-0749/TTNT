"""Eye tracking, blink counting and sleepiness detection from face landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from time import monotonic
from typing import Any, Iterable, Mapping, Sequence


Point = tuple[float, float]


@dataclass(frozen=True)
class EyeMetrics:
    """Measurements for one analyzed frame."""

    left_ear: float
    right_ear: float
    average_ear: float
    left_state: str
    right_state: str
    eye_state: str
    blink_count: int
    closure_duration: float
    is_sleepy: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable result for the other application modules."""
        return {
            "left_ear": round(self.left_ear, 4),
            "right_ear": round(self.right_ear, 4),
            "average_ear": round(self.average_ear, 4),
            "left_state": self.left_state,
            "right_state": self.right_state,
            "eye_state": self.eye_state,
            "blink_count": self.blink_count,
            "closure_duration": round(self.closure_duration, 3),
            "is_sleepy": self.is_sleepy,
        }


class EyeAnalyzer:
    """Analyze MediaPipe Face Mesh landmarks for eye activity.

    Landmark order follows MediaPipe Face Mesh. Coordinates may be normalized
    (0..1), pixel coordinates, or landmark objects exposing ``x`` and ``y``.
    """

    LEFT_EYE = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE = (362, 385, 387, 263, 373, 380)

    def __init__(
        self,
        ear_threshold: float = 0.21,
        sleepy_duration: float = 2.0,
        max_blink_duration: float = 0.8,
    ) -> None:
        if not 0 < ear_threshold < 1:
            raise ValueError("ear_threshold must be between 0 and 1")
        if sleepy_duration <= 0 or max_blink_duration <= 0:
            raise ValueError("duration thresholds must be positive")

        self.ear_threshold = ear_threshold
        self.sleepy_duration = sleepy_duration
        self.max_blink_duration = max_blink_duration
        self.blink_count = 0
        self._closed_since: float | None = None
        self._was_closed = False
        self._last_timestamp: float | None = None

    @staticmethod
    def eye_aspect_ratio(eye_points: Sequence[Point]) -> float:
        """Calculate EAR from [outer, upper-1, upper-2, inner, lower-2, lower-1]."""
        if len(eye_points) != 6:
            raise ValueError("an eye must contain exactly 6 landmarks")

        outer, upper_one, upper_two, inner, lower_two, lower_one = eye_points
        horizontal = hypot(inner[0] - outer[0], inner[1] - outer[1])
        if horizontal == 0:
            return 0.0
        vertical_one = hypot(lower_one[0] - upper_one[0], lower_one[1] - upper_one[1])
        vertical_two = hypot(lower_two[0] - upper_two[0], lower_two[1] - upper_two[1])
        return (vertical_one + vertical_two) / (2.0 * horizontal)

    @staticmethod
    def _point(landmark: Any) -> Point:
        if hasattr(landmark, "x") and hasattr(landmark, "y"):
            return float(landmark.x), float(landmark.y)
        if isinstance(landmark, Mapping):
            return float(landmark["x"]), float(landmark["y"])
        return float(landmark[0]), float(landmark[1])

    def _eye_points(self, landmarks: Sequence[Any], indices: Sequence[int]) -> list[Point]:
        if len(landmarks) <= max(indices):
            raise ValueError("landmarks must contain the 468 MediaPipe face points")
        return [self._point(landmarks[index]) for index in indices]

    def reset(self) -> None:
        """Reset blink and closure history, for example when the face changes."""
        self.blink_count = 0
        self._closed_since = None
        self._was_closed = False
        self._last_timestamp = None

    def process_landmarks(
        self,
        landmarks: Iterable[Any],
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """Analyze one face and return metrics as a dictionary.

        ``timestamp`` is measured in seconds and should be monotonic. If it is
        omitted, the system clock is used. A single blink is counted when both
        eyes close and reopen within ``max_blink_duration``.
        """
        landmark_list = list(landmarks)
        left_ear = self.eye_aspect_ratio(self._eye_points(landmark_list, self.LEFT_EYE))
        right_ear = self.eye_aspect_ratio(self._eye_points(landmark_list, self.RIGHT_EYE))
        average_ear = (left_ear + right_ear) / 2.0
        now = monotonic() if timestamp is None else float(timestamp)
        if self._last_timestamp is not None and now < self._last_timestamp:
            raise ValueError("timestamp must be monotonic")
        self._last_timestamp = now

        left_closed = left_ear < self.ear_threshold
        right_closed = right_ear < self.ear_threshold
        both_closed = left_closed and right_closed
        if both_closed and not self._was_closed:
            self._closed_since = now
        elif not both_closed and self._was_closed and self._closed_since is not None:
            if now - self._closed_since <= self.max_blink_duration:
                self.blink_count += 1
            self._closed_since = None
        self._was_closed = both_closed

        closure_duration = 0.0
        if both_closed and self._closed_since is not None:
            closure_duration = max(0.0, now - self._closed_since)

        return EyeMetrics(
            left_ear=left_ear,
            right_ear=right_ear,
            average_ear=average_ear,
            left_state="CLOSED" if left_closed else "OPEN",
            right_state="CLOSED" if right_closed else "OPEN",
            eye_state="CLOSED" if both_closed else "OPEN",
            blink_count=self.blink_count,
            closure_duration=closure_duration,
            is_sleepy=both_closed and closure_duration >= self.sleepy_duration,
        ).as_dict()