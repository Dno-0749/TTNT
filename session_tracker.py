"""
============================================================
 SESSION TRACKER
============================================================
Module          : utils/session_tracker.py
Phụ trách       : Person 5 — Focus Engine + Temporal Analysis

MÔ TẢ (dùng để thuyết trình)
------------------------------------------------------------
Nhận luồng FocusResult do FocusEngine sinh ra mỗi frame (xem
modules/focus_engine.py) và cộng dồn thành số liệu của toàn bộ
phiên học (session):

    - Study Time        : tổng thời gian của cả phiên
    - Focused Time       : thời gian ở trạng thái FOCUSED
    - Distracted Time    : thời gian ở trạng thái DISTRACTED hoặc WARNING
                            (WARNING là mức "mất tập trung nghiêm trọng",
                             vẫn được tính vào Distracted Time; số lần vào
                             WARNING được đếm riêng ở Warning Count)
    - Sleepy Time        : thời gian ở trạng thái SLEEPY
    - Warning Count      : tổng số lần chuyển sang trạng thái WARNING

Cuối phiên, generate_report() / print_report() / export_json() xuất
báo cáo tổng kết — dùng để hiển thị lên UI hoặc làm số liệu minh hoạ
cho slide thuyết trình.
============================================================
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from modules.focus_engine import FocusResult, FocusState
except ImportError:
    # Cho phép chạy độc lập file này khi chưa gắn vào cấu trúc package đầy đủ
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from modules.focus_engine import FocusResult, FocusState


# ============================================================
# 1. SESSION STATS — số liệu thô cộng dồn theo thời gian
# ============================================================

@dataclass
class SessionStats:
    study_time_sec: float = 0.0
    focused_time_sec: float = 0.0
    distracted_time_sec: float = 0.0
    sleepy_time_sec: float = 0.0
    warning_count: int = 0

    frame_count: int = 0
    focus_score_sum: float = 0.0

    @property
    def average_focus_score(self) -> float:
        return round(self.focus_score_sum / self.frame_count, 2) if self.frame_count else 0.0

    @property
    def focused_ratio(self) -> float:
        return round(100 * self.focused_time_sec / self.study_time_sec, 1) if self.study_time_sec else 0.0

    @property
    def distracted_ratio(self) -> float:
        return round(100 * self.distracted_time_sec / self.study_time_sec, 1) if self.study_time_sec else 0.0

    @property
    def sleepy_ratio(self) -> float:
        return round(100 * self.sleepy_time_sec / self.study_time_sec, 1) if self.study_time_sec else 0.0


# ============================================================
# 2. SESSION TRACKER
# ============================================================

class SessionTracker:
    """
    Sử dụng:
        tracker = SessionTracker(user_id="sv001", session_name="Buổi học Toán")
        for frame in stream_of_frames:
            result = focus_engine.process(frame)
            tracker.update(result)
        tracker.end()
        tracker.print_report()
        tracker.export_json("report.json")
    """

    def __init__(self, user_id: str = "unknown", session_name: str = "Study Session"):
        self.user_id = user_id
        self.session_name = session_name
        self.stats = SessionStats()

        self.state_log: List[Dict] = []  # log mỗi lần trạng thái thay đổi

        self._last_result: Optional[FocusResult] = None
        self._last_timestamp: Optional[float] = None
        self._session_start: Optional[float] = None
        self._session_end: Optional[float] = None
        self._warning_seen: int = 0

    # ---- vòng đời phiên học ----

    def start(self, timestamp: Optional[float] = None) -> None:
        self._session_start = timestamp if timestamp is not None else time.time()
        self._last_timestamp = self._session_start

    def update(self, result: FocusResult) -> None:
        """Gọi hàm này mỗi khi FocusEngine trả về một FocusResult (mỗi frame)."""
        if self._session_start is None:
            self.start(result.timestamp)

        dt = 0.0
        if self._last_timestamp is not None:
            dt = max(0.0, result.timestamp - self._last_timestamp)
        self._last_timestamp = result.timestamp

        # ---- cộng dồn thời gian theo trạng thái hiện tại ----
        s = self.stats
        s.study_time_sec += dt
        if result.state == FocusState.FOCUSED:
            s.focused_time_sec += dt
        elif result.state in (FocusState.DISTRACTED, FocusState.WARNING):
            s.distracted_time_sec += dt
        elif result.state == FocusState.SLEEPY:
            s.sleepy_time_sec += dt

        # ---- warning count lấy trực tiếp từ FocusEngine (đã đếm theo transition) ----
        if result.warning_count_total > self._warning_seen:
            self._warning_seen = result.warning_count_total
        s.warning_count = self._warning_seen

        s.frame_count += 1
        s.focus_score_sum += result.focus_score

        # ---- log transition để dựng timeline trong báo cáo ----
        if self._last_result is None or result.state != self._last_result.state:
            self.state_log.append(
                {
                    "timestamp": result.timestamp,
                    "state": result.state.value,
                    "focus_score": result.focus_score,
                }
            )

        self._last_result = result

    def end(self, timestamp: Optional[float] = None) -> None:
        self._session_end = timestamp if timestamp is not None else (self._last_timestamp or time.time())

    # ---- báo cáo ----

    def generate_report(self) -> Dict:
        r = self.stats
        return {
            "user_id": self.user_id,
            "session_name": self.session_name,
            "start_time": self._session_start,
            "end_time": self._session_end,
            "study_time_sec": round(r.study_time_sec, 1),
            "focused_time_sec": round(r.focused_time_sec, 1),
            "distracted_time_sec": round(r.distracted_time_sec, 1),
            "sleepy_time_sec": round(r.sleepy_time_sec, 1),
            "warning_count": r.warning_count,
            "average_focus_score": r.average_focus_score,
            "focused_ratio_percent": r.focused_ratio,
            "distracted_ratio_percent": r.distracted_ratio,
            "sleepy_ratio_percent": r.sleepy_ratio,
            "state_transitions": self.state_log,
        }

    def print_report(self) -> None:
        r = self.generate_report()
        line = "=" * 52
        print(line)
        print(f" BÁO CÁO PHIÊN HỌC — {r['session_name']}")
        print(line)
        print(f" Người dùng               : {r['user_id']}")
        print(f" Tổng thời gian học        : {self._format_time(r['study_time_sec'])}")
        print(
            f" Thời gian tập trung       : {self._format_time(r['focused_time_sec'])} "
            f"({r['focused_ratio_percent']}%)"
        )
        print(
            f" Thời gian mất tập trung   : {self._format_time(r['distracted_time_sec'])} "
            f"({r['distracted_ratio_percent']}%)"
        )
        print(
            f" Thời gian buồn ngủ        : {self._format_time(r['sleepy_time_sec'])} "
            f"({r['sleepy_ratio_percent']}%)"
        )
        print(f" Số lần cảnh báo (Warning) : {r['warning_count']}")
        print(f" Điểm tập trung trung bình : {r['average_focus_score']}/100")
        print(line)

    def export_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.generate_report(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _format_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


# ============================================================
# 3. DEMO END-TO-END: FocusEngine + SessionTracker
# ============================================================

if __name__ == "__main__":
    from modules.focus_engine import (
        EyeData,
        FaceData,
        FocusEngine,
        FrameInput,
        HeadPoseData,
        MLPrediction,
    )

    engine = FocusEngine()
    tracker = SessionTracker(user_id="sv001", session_name="Demo buổi học 12 giây")

    scenario = [
        (0, 0, True, True, 0.30, "focused", 0.9, 0.0),
        (5, 0, True, True, 0.30, "focused", 0.9, 1.0),
        (40, 5, True, False, 0.30, "distracted", 0.8, 1.0),
        (40, 5, True, False, 0.30, "distracted", 0.8, 1.0),
        (40, 5, True, False, 0.30, "distracted", 0.8, 1.0),
        (40, 5, True, False, 0.30, "distracted", 0.8, 1.0),
        (40, 5, True, False, 0.30, "distracted", 0.8, 1.0),
        (0, 0, True, True, 0.30, "focused", 0.9, 1.0),
        (0, 0, True, True, 0.30, "focused", 0.9, 1.0),
        (0, 0, False, True, 0.10, "sleepy", 0.9, 1.0),
        (0, 0, False, True, 0.10, "sleepy", 0.9, 1.0),
        (0, 0, False, True, 0.10, "sleepy", 0.9, 1.0),
        (0, 0, False, True, 0.10, "sleepy", 0.9, 1.0),
    ]

    t = 0.0
    for yaw, pitch, eyes_open, gaze_on_screen, ear, ml_label, ml_conf, dt in scenario:
        t += dt
        frame = FrameInput(
            face=FaceData(face_detected=True, confidence=0.95),
            eye=EyeData(eyes_open=eyes_open, gaze_on_screen=gaze_on_screen, eye_aspect_ratio=ear),
            head_pose=HeadPoseData(yaw=yaw, pitch=pitch),
            ml=MLPrediction(label=ml_label, confidence=ml_conf),
            timestamp=t,
        )
        result = engine.process(frame)
        tracker.update(result)

    tracker.end()
    tracker.print_report()

    out_path = "/tmp/session_report_demo.json"
    tracker.export_json(out_path)
    print(f"\n(Đã xuất báo cáo JSON demo tại: {out_path})")
