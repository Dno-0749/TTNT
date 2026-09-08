"""
============================================================
 FOCUS ENGINE + TEMPORAL ANALYSIS
============================================================
Module          : modules/focus_engine.py
Phụ trách       : Person 5 — Focus Engine + Temporal Analysis

MÔ TẢ THUẬT TOÁN (dùng để thuyết trình)
------------------------------------------------------------
Input mỗi frame gồm 4 nguồn:
    Face (phát hiện khuôn mặt) + Eye (mắt) + Head Pose (góc quay đầu)
    + ML prediction (model phân loại attentive/distracted/sleepy)

Pipeline xử lý:
    FrameInput
        │
        ▼
    FocusScoreCalculator   -> tính 4 thành phần điểm (0..1) + Focus Score (0..100)
        │
        ▼
    TemporalAnalyzer       -> đo thời gian liên tục của hành vi mất tập trung /
                               nhắm mắt để "đề xuất" trạng thái tiếp theo
        │
        ▼
    FocusStateMachine      -> xác nhận transition, ghi log, đếm Warning Count
        │
        ▼
    FocusResult (trả về cho session_tracker.py mỗi frame)

1) FOCUS SCORE (trọng số theo đúng thiết kế của nhóm)
    Head Orientation   40%
    Eye Attention      30%
    Face Presence      20%
    Continuity         10%
    => Focus Score = (0.4*Head + 0.3*Eye + 0.2*Face + 0.1*Continuity) * 100

2) TEMPORAL ANALYSIS (ví dụ minh hoạ của nhóm)
    Quay đầu 1s  -> bình thường   (< 2s)
    Quay đầu 3s  -> DISTRACTED    (>= 2s, < 4s)
    Quay đầu 5s  -> WARNING       (>= 4s)
    Nhắm mắt liên tục >= 3s -> SLEEPY (kênh riêng, ưu tiên cao nhất)

3) STATE MACHINE
    FOCUSED -> DISTRACTED -> WARNING -> FOCUSED
    FOCUSED -> SLEEPY -> FOCUSED   (nhánh riêng, kích hoạt khi nhắm mắt lâu)
============================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from typing import Deque, Optional, List, Dict, Any, Tuple


# ============================================================
# 0. CẤU HÌNH — mọi ngưỡng/threshold tập trung tại đây để dễ tinh chỉnh
#    khi demo hoặc khi thầy/cô yêu cầu đổi thông số.
# ============================================================

@dataclass
class FocusEngineConfig:
    # ---- Trọng số tính Focus Score (phải cộng lại = 1.0) ----
    WEIGHT_HEAD: float = 0.40
    WEIGHT_EYE: float = 0.30
    WEIGHT_FACE: float = 0.20
    WEIGHT_CONTINUITY: float = 0.10

    # ---- Ngưỡng góc quay đầu (độ) ----
    HEAD_YAW_SAFE_DEG: float = 10.0     # trong khoảng này coi như vẫn nhìn thẳng màn hình
    HEAD_YAW_MAX_DEG: float = 35.0      # vượt quá góc này -> điểm head = 0
    HEAD_PITCH_SAFE_DEG: float = 8.0
    HEAD_PITCH_MAX_DEG: float = 25.0

    # ---- Ngưỡng mắt (Eye Aspect Ratio) ----
    EAR_OPEN_THRESHOLD: float = 0.21    # EAR thấp hơn ngưỡng này -> coi như nhắm mắt

    # ---- Continuity: cửa sổ trượt đánh giá độ ổn định ----
    CONTINUITY_WINDOW_SIZE: int = 30    # số frame gần nhất (≈1s nếu chạy ở 30fps)

    # ---- Temporal thresholds (giây) — đúng theo ví dụ minh hoạ của nhóm ----
    DISTRACTED_AFTER_SEC: float = 2.0
    WARNING_AFTER_SEC: float = 4.0
    SLEEPY_AFTER_SEC: float = 3.0

    # ---- Ngưỡng điểm để 1 frame được coi là "đang tập trung" tức thời ----
    ATTENTIVE_SCORE_THRESHOLD: float = 60.0  # thang 0..100


# ============================================================
# 1. DATA MODELS — input từ các module khác (Face / Eye / Head Pose / ML)
# ============================================================

@dataclass
class FaceData:
    """Kết quả từ module phát hiện khuôn mặt."""
    face_detected: bool
    confidence: float = 1.0            # 0.0 - 1.0


@dataclass
class EyeData:
    """Kết quả từ module theo dõi mắt."""
    eyes_open: bool
    gaze_on_screen: bool = True        # hướng nhìn có đang tập trung vào màn hình
    eye_aspect_ratio: float = 0.30     # EAR — dùng để phát hiện buồn ngủ


@dataclass
class HeadPoseData:
    """Góc quay đầu ước lượng, đơn vị độ (degree)."""
    yaw: float = 0.0     # quay trái/phải
    pitch: float = 0.0   # ngẩng/cúi
    roll: float = 0.0    # nghiêng đầu (hiện chưa dùng để tính điểm, để mở rộng sau)


@dataclass
class MLPrediction:
    """Kết quả dự đoán từ model ML/DL (đóng vai trò cross-check)."""
    label: str = "focused"   # "focused" | "distracted" | "sleepy"
    confidence: float = 1.0  # 0.0 - 1.0


@dataclass
class FrameInput:
    """Gói toàn bộ input của một frame."""
    face: FaceData
    eye: EyeData
    head_pose: HeadPoseData
    ml: MLPrediction
    timestamp: float = field(default_factory=time.time)


# ============================================================
# 2. TRẠNG THÁI TẬP TRUNG
# ============================================================

class FocusState(Enum):
    FOCUSED = "FOCUSED"
    DISTRACTED = "DISTRACTED"
    WARNING = "WARNING"
    SLEEPY = "SLEEPY"


@dataclass
class FocusResult:
    """Kết quả trả ra mỗi frame — session_tracker.py sẽ tiêu thụ dữ liệu này."""
    timestamp: float
    focus_score: float                  # 0..100
    state: FocusState
    components: Dict[str, float]        # head / eye / face / continuity / ml_adjustment
    distraction_duration: float         # số giây đang mất tập trung liên tục
    eyes_closed_duration: float         # số giây đang nhắm mắt liên tục
    warning_count_total: int            # tổng số lần đã vào WARNING tính đến frame này


# ============================================================
# 3. FOCUS SCORE CALCULATOR
# ============================================================

class FocusScoreCalculator:
    """
    Tính Focus Score từ 4 thành phần: Head Orientation, Eye Attention,
    Face Presence, Continuity — có tích hợp thêm hiệu chỉnh từ ML prediction
    như một lớp kiểm tra chéo (cross-check).
    """

    def __init__(self, config: FocusEngineConfig):
        self.config = config
        # lưu lại các điểm "tức thời" (chưa gồm continuity) để tính continuity
        self._history: Deque[float] = deque(maxlen=config.CONTINUITY_WINDOW_SIZE)

    # ---- các hàm điểm thành phần ----

    @staticmethod
    def _axis_falloff(angle: float, safe_deg: float, max_deg: float) -> float:
        """
        Hàm suy giảm tuyến tính: 1.0 trong vùng an toàn, giảm dần về 0.0
        khi góc tiến tới ngưỡng tối đa.
        """
        a = abs(angle)
        if a <= safe_deg:
            return 1.0
        if a >= max_deg:
            return 0.0
        return 1.0 - (a - safe_deg) / (max_deg - safe_deg)

    def _head_orientation_score(self, pose: HeadPoseData) -> float:
        c = self.config
        yaw_score = self._axis_falloff(pose.yaw, c.HEAD_YAW_SAFE_DEG, c.HEAD_YAW_MAX_DEG)
        pitch_score = self._axis_falloff(pose.pitch, c.HEAD_PITCH_SAFE_DEG, c.HEAD_PITCH_MAX_DEG)
        # yaw (quay trái/phải) thể hiện rõ việc "nhìn ra ngoài màn hình" hơn pitch
        # nên cho trọng số cao hơn một chút trong nội bộ thành phần head.
        return 0.6 * yaw_score + 0.4 * pitch_score

    def _eye_attention_score(self, eye: EyeData) -> float:
        c = self.config
        if not eye.eyes_open or eye.eye_aspect_ratio < c.EAR_OPEN_THRESHOLD:
            return 0.0        # mắt nhắm -> không tập trung (đồng thời là tín hiệu buồn ngủ)
        if not eye.gaze_on_screen:
            return 0.35       # mắt mở nhưng ánh nhìn lệch khỏi màn hình
        return 1.0

    def _face_presence_score(self, face: FaceData) -> float:
        if not face.face_detected:
            return 0.0
        return max(0.0, min(1.0, face.confidence))

    def _ml_adjustment(self, ml: MLPrediction) -> float:
        """
        ML prediction đóng vai trò lớp kiểm tra chéo bên cạnh phần rule-based
        (head pose + eye). Nếu model tự tin dự đoán "distracted"/"sleepy" dù
        tín hiệu hình học vẫn ổn, ta trừ điểm; nếu ML xác nhận "focused",
        cộng nhẹ để tăng độ tin cậy tổng thể.
        """
        if ml.label == "focused":
            return 0.03 * ml.confidence
        if ml.label == "distracted":
            return -0.10 * ml.confidence
        if ml.label == "sleepy":
            return -0.15 * ml.confidence
        return 0.0

    # ---- hàm chính ----

    def compute(self, frame: FrameInput) -> Tuple[Dict[str, float], float]:
        c = self.config

        head_score = self._head_orientation_score(frame.head_pose)
        eye_score = self._eye_attention_score(frame.eye)
        face_score = self._face_presence_score(frame.face)

        # điểm tức thời (thang 0 .. 0.9, do chưa cộng phần continuity 10%)
        max_instant = c.WEIGHT_HEAD + c.WEIGHT_EYE + c.WEIGHT_FACE  # = 0.9
        instantaneous = (
            c.WEIGHT_HEAD * head_score
            + c.WEIGHT_EYE * eye_score
            + c.WEIGHT_FACE * face_score
        )

        ml_adj = self._ml_adjustment(frame.ml)
        instantaneous = max(0.0, min(max_instant, instantaneous + ml_adj))

        # continuity = mức ổn định trung bình của các frame gần đây
        # (giúp "phạt" tình trạng chập chờn / to-in-out-of-focus liên tục)
        self._history.append(instantaneous)
        continuity_raw = sum(self._history) / len(self._history) / max_instant
        continuity_score = max(0.0, min(1.0, continuity_raw))

        final_fraction = instantaneous + c.WEIGHT_CONTINUITY * continuity_score
        final_score_100 = round(final_fraction * 100, 2)

        components = {
            "head_orientation": round(head_score, 3),
            "eye_attention": round(eye_score, 3),
            "face_presence": round(face_score, 3),
            "continuity": round(continuity_score, 3),
            "ml_adjustment": round(ml_adj, 3),
        }
        return components, final_score_100


# ============================================================
# 4. TEMPORAL ANALYZER
# ============================================================

class TemporalAnalyzer:
    """
    Theo dõi độ dài liên tục (duration) của 2 loại hành vi:
        - "mất tập trung" (is_attentive == False)
        - "nhắm mắt"      (eyes_closed == True)
    và quy đổi ra trạng thái đề xuất (proposed state) dựa trên các
    ngưỡng thời gian đã cấu hình.
    """

    def __init__(self, config: FocusEngineConfig):
        self.config = config
        self._distraction_start: Optional[float] = None
        self._eyes_closed_start: Optional[float] = None

    def analyze(
        self, timestamp: float, is_attentive: bool, eyes_closed: bool
    ) -> Tuple[FocusState, float, float]:
        c = self.config

        # ---- kênh buồn ngủ (ưu tiên cao nhất) ----
        if eyes_closed:
            if self._eyes_closed_start is None:
                self._eyes_closed_start = timestamp
            eyes_closed_duration = timestamp - self._eyes_closed_start
        else:
            self._eyes_closed_start = None
            eyes_closed_duration = 0.0

        if eyes_closed_duration >= c.SLEEPY_AFTER_SEC:
            # buồn ngủ được ưu tiên hơn, reset timer mất tập trung để tránh
            # 2 timer cộng dồn chồng chéo nhau khi quay lại trạng thái bình thường
            self._distraction_start = None
            return FocusState.SLEEPY, 0.0, eyes_closed_duration

        # ---- kênh mất tập trung (quay đầu / gaze lệch / điểm thấp) ----
        if not is_attentive:
            if self._distraction_start is None:
                self._distraction_start = timestamp
            distraction_duration = timestamp - self._distraction_start
        else:
            self._distraction_start = None
            distraction_duration = 0.0

        if distraction_duration >= c.WARNING_AFTER_SEC:
            state = FocusState.WARNING
        elif distraction_duration >= c.DISTRACTED_AFTER_SEC:
            state = FocusState.DISTRACTED
        else:
            state = FocusState.FOCUSED

        return state, distraction_duration, eyes_closed_duration


# ============================================================
# 5. STATE MACHINE
# ============================================================

class FocusStateMachine:
    """
    Xác nhận transition do TemporalAnalyzer đề xuất, ghi log lịch sử
    transition (để vẽ timeline trong báo cáo/slide), và đếm số lần
    đi vào trạng thái WARNING (Warning Count).

        FOCUSED -> DISTRACTED -> WARNING -> FOCUSED
        FOCUSED -> SLEEPY -> FOCUSED
    """

    def __init__(self):
        self.current_state: FocusState = FocusState.FOCUSED
        self.history: List[Dict[str, Any]] = []
        self.warning_count: int = 0

    def transition(self, proposed_state: FocusState, timestamp: float) -> FocusState:
        if proposed_state != self.current_state:
            self.history.append(
                {
                    "from": self.current_state.value,
                    "to": proposed_state.value,
                    "timestamp": timestamp,
                }
            )
            if proposed_state == FocusState.WARNING:
                self.warning_count += 1
            self.current_state = proposed_state
        return self.current_state


# ============================================================
# 6. FOCUS ENGINE — orchestrator chính, được gọi mỗi frame
# ============================================================

class FocusEngine:
    """
    Điểm vào (entry point) chính của module. Mỗi frame gọi:
        result = engine.process(frame_input)
    và đẩy `result` sang session_tracker.SessionTracker.update(result).
    """

    def __init__(self, config: Optional[FocusEngineConfig] = None):
        self.config = config or FocusEngineConfig()
        self._score_calc = FocusScoreCalculator(self.config)
        self._temporal = TemporalAnalyzer(self.config)
        self._state_machine = FocusStateMachine()

    def process(self, frame: FrameInput) -> FocusResult:
        c = self.config
        components, focus_score = self._score_calc.compute(frame)

        is_attentive = focus_score >= c.ATTENTIVE_SCORE_THRESHOLD
        eyes_closed = (
            not frame.eye.eyes_open
            or frame.eye.eye_aspect_ratio < c.EAR_OPEN_THRESHOLD
            or (frame.ml.label == "sleepy" and frame.ml.confidence >= 0.6)
        )

        proposed_state, distraction_duration, eyes_closed_duration = self._temporal.analyze(
            frame.timestamp, is_attentive, eyes_closed
        )
        final_state = self._state_machine.transition(proposed_state, frame.timestamp)

        return FocusResult(
            timestamp=frame.timestamp,
            focus_score=focus_score,
            state=final_state,
            components=components,
            distraction_duration=round(distraction_duration, 2),
            eyes_closed_duration=round(eyes_closed_duration, 2),
            warning_count_total=self._state_machine.warning_count,
        )

    @property
    def transition_history(self) -> List[Dict[str, Any]]:
        return self._state_machine.history


# ============================================================
# 7. DEMO CHẠY THỬ (dùng để test nhanh / chụp màn hình cho slide)
# ============================================================

if __name__ == "__main__":
    engine = FocusEngine()

    # Kịch bản demo: tập trung -> quay đầu tăng dần -> quay lại tập trung
    # -> nhắm mắt kéo dài (buồn ngủ). timestamps mô phỏng theo giây thực.
    scenario = [
        # (yaw, pitch, eyes_open, gaze_on_screen, ear, ml_label, ml_conf, dt)
        (0, 0, True, True, 0.30, "focused", 0.9, 0.0),
        (5, 0, True, True, 0.30, "focused", 0.9, 1.0),
        (40, 5, True, False, 0.30, "distracted", 0.8, 1.0),  # bắt đầu quay đầu (t=1s)
        (40, 5, True, False, 0.30, "distracted", 0.8, 1.0),  # t=2s -> DISTRACTED
        (40, 5, True, False, 0.30, "distracted", 0.8, 1.0),  # t=3s
        (40, 5, True, False, 0.30, "distracted", 0.8, 1.0),  # t=4s -> WARNING
        (40, 5, True, False, 0.30, "distracted", 0.8, 1.0),  # t=5s
        (0, 0, True, True, 0.30, "focused", 0.9, 1.0),       # quay lại tập trung
        (0, 0, True, True, 0.30, "focused", 0.9, 1.0),
        (0, 0, False, True, 0.10, "sleepy", 0.9, 1.0),       # bắt đầu nhắm mắt (duration=0s)
        (0, 0, False, True, 0.10, "sleepy", 0.9, 1.0),       # duration=1s
        (0, 0, False, True, 0.10, "sleepy", 0.9, 1.0),       # duration=2s
        (0, 0, False, True, 0.10, "sleepy", 0.9, 1.0),       # duration=3s -> SLEEPY
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
        print(
            f"t={t:5.1f}s | Score={result.focus_score:6.2f} | State={result.state.value:11s} "
            f"| distract_dur={result.distraction_duration:4.1f}s | eyes_closed_dur={result.eyes_closed_duration:4.1f}s "
            f"| warnings={result.warning_count_total}"
        )

    print("\n--- Lịch sử transition ---")
    for h in engine.transition_history:
        print(h)
