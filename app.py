import os
import sys

# Thêm đường dẫn gốc để đảm bảo Python nhận diện đúng mọi module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from time import time, sleep
import cv2
import streamlit as st

from data.data_manager import get_data, update_data, reset_session_data

from dashboard.realtime import (
    show_realtime,
    show_statistics,
    calculate_realtime
)

from dashboard.analytics import (
    show_analytics,
    show_report,
    show_recommendations
)

# Import các module cốt lõi
from modules.camera import Camera
from modules.head_pose import HeadPoseEstimator
from modules.eye_analyzer import EyeAnalyzer
from modules.focus_engine import FocusEngine
from modules.session_tracker import SessionTracker


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="SMARTSTUDY AI",
    layout="wide"
)

st.title("🎓 SMARTSTUDY AI")
st.write("AI Student Focus Monitoring System")

st.divider()


# =========================
# SESSION STATE INITIALIZATION
# =========================

if "warning_count" not in st.session_state:
    st.session_state.warning_count = 0

if "prev_warning_state" not in st.session_state:
    st.session_state.prev_warning_state = False

if "tracker" not in st.session_state:
    st.session_state.tracker = SessionTracker()

if "focus_engine" not in st.session_state:
    st.session_state.focus_engine = FocusEngine()

if "eye_analyzer" not in st.session_state:
    st.session_state.eye_analyzer = EyeAnalyzer()


# =========================
# SIDEBAR CONTROLS
# =========================

with st.sidebar:
    st.header("⚙️ Cấu Hình & Điều Khiển")

    camera_idx = st.selectbox("📷 Chọn Camera Index", options=[0, 1, 2], index=0)

    camera_width = st.slider(
        "📐 Chiều rộng camera",
        min_value=320,
        max_value=960,
        value=640,
        step=40
    )
    camera_height = int(camera_width * 480 / 640)
    st.caption(f"Độ phân giải: {camera_width} × {camera_height}")

    st.divider()

    st.subheader("🔄 Quản lý Phiên học")
    if st.button("Làm mới Phiên học (Reset Session)", use_container_width=True):
        st.session_state.tracker.reset()
        st.session_state.focus_engine.reset()
        st.session_state.eye_analyzer.reset()
        st.session_state.warning_count = 0
        st.session_state.prev_warning_state = False
        reset_session_data()
        st.success("Đã đặt lại dữ liệu phiên học!")
        st.rerun()

    st.divider()
    st.caption("🤖 Model: Machine Learning Classification")
    st.caption("👁️ Eye Tracking: MediaPipe Face Mesh (EAR)")
    st.caption("🗣️ Head Pose: 3D SolvePnP (Yaw/Pitch/Roll)")


# =========================
# CUSTOM CSS STYLING
# =========================

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 12px 16px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    div[data-testid="stImage"] img {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# MAIN DASHBOARD LAYOUT (2 COLUMNS)
# =========================

col_cam, col_hud = st.columns([1.15, 0.85], gap="large")

with col_cam:
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">📷 Luồng Giám Sát Camera</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    frame_placeholder = st.empty()

    # Thanh điều khiển Camera gọn gàng
    ctrl_col1, ctrl_col2 = st.columns([1.2, 1])
    with ctrl_col1:
        start_camera = st.toggle("🎥 Kích hoạt Camera Giám sát", value=False, key="camera_toggle")
    with ctrl_col2:
        with st.popover("⚙️ Cài đặt Camera"):
            camera_idx = st.selectbox("Cổng Camera (Index)", options=[0, 1, 2], index=0)
            camera_width = st.slider("Chiều rộng hiển thị (px)", 360, 960, 640, 40)
            camera_height = int(camera_width * 480 / 640)
            st.caption(f"Độ phân giải: {camera_width} × {camera_height}")

with col_hud:
    st.markdown(
        """
        <div style="margin-bottom: 8px;">
            <span style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">📊 Chỉ Số Thời Gian Thực</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    realtime_placeholder = st.container()


# =========================
# CAMERA EXECUTION OR STANDBY
# =========================

if not start_camera:
    with frame_placeholder:
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.35); border: 2px dashed rgba(255,255,255,0.14); border-radius: 16px; padding: 75px 20px; text-align: center; margin: 4px 0 14px 0;">
                <div style="font-size: 3.5rem; margin-bottom: 8px;">📹</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #f1f5f9;">Camera Đang Ở Trạng Thái Chờ</div>
                <div style="font-size: 0.88rem; color: #94a3b8; max-width: 400px; margin: 8px auto 0 auto; line-height: 1.5;">
                    Gạt nút <b>"Kích hoạt Camera Giám sát"</b> bên dưới để AI tự động phân tích hướng nhìn và mức độ tập trung.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with realtime_placeholder:
        show_realtime(get_data())

else:
    # Khởi tạo Camera và HeadPoseEstimator
    camera = Camera(
        camera_index=camera_idx,
        width=camera_width,
        height=camera_height
    )

    estimator = HeadPoseEstimator()

    last_ui_update_time = 0.0
    last_ui_state = ""

    try:
        while start_camera:

            # 1. Lấy frame từ camera
            frame = camera.get_frame()

            if frame is None:
                st.error("❌ Không thể đọc luồng dữ liệu từ camera. Vui lòng kiểm tra lại webcam.")
                break

            # 2. Xử lý AI Head Pose & Tọa độ Landmarks
            annotated_frame, feature_dict = estimator.process_frame(frame)
            raw_landmarks = feature_dict.get("raw_landmarks")
            face_detected = feature_dict.get("face_detected", False)
            face_count = feature_dict.get("face_count", 1 if face_detected else 0)

            # 3. Phân tích trạng thái mắt qua EyeAnalyzer (EAR, chớp mắt, buồn ngủ)
            if raw_landmarks is not None:
                eye_metrics = st.session_state.eye_analyzer.process_landmarks(raw_landmarks)
            else:
                if not face_detected:
                    st.session_state.eye_analyzer.reset()
                eye_metrics = {
                    "left_ear": 0.0,
                    "right_ear": 0.0,
                    "average_ear": 0.0,
                    "left_state": "UNKNOWN",
                    "right_state": "UNKNOWN",
                    "eye_state": "UNKNOWN" if not face_detected else "OPEN",
                    "blink_count": st.session_state.eye_analyzer.blink_count,
                    "closure_duration": 0.0,
                    "is_sleepy": False
                }

            ear = eye_metrics["average_ear"]
            blink_count = eye_metrics["blink_count"]
            closure_duration = eye_metrics["closure_duration"]
            yaw = feature_dict.get("yaw", 0.0)
            pitch = feature_dict.get("pitch", 0.0)
            roll = feature_dict.get("roll", 0.0)

            # Vẽ thông số mắt trực tiếp lên khung hình camera
            if face_detected:
                eye_color = (80, 210, 120) if eye_metrics["eye_state"] == "OPEN" else (30, 80, 255)
                # Mini HUD mắt ở góc dưới trái của video
                cv2.putText(
                    annotated_frame,
                    f"EAR: {ear:.2f} ({eye_metrics['eye_state']}) | Chop mat: {blink_count}",
                    (20, camera_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    eye_color,
                    2,
                    cv2.LINE_AA
                )
                if eye_metrics["is_sleepy"]:
                    cv2.putText(
                        annotated_frame,
                        "CANH BAO: NGU GAT / MAT NHAM QUA LAU!",
                        (20, camera_height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (30, 80, 255),
                        2,
                        cv2.LINE_AA
                    )

            # 4. Dự đoán trạng thái bằng Model ML & Tính điểm Focus Engine
            ml_state = st.session_state.focus_engine.predict_state(
                face_count=face_count,
                ear=ear,
                blink_count=blink_count,
                closure_duration=closure_duration,
                yaw=yaw,
                pitch=pitch,
                roll=roll
            )

            focus_score, final_state = st.session_state.focus_engine.calculate_score(
                eye_data=eye_metrics,
                head_yaw=yaw,
                face_count=face_count,
                ml_state=ml_state
            )

            # 5. Cập nhật Session Tracker
            st.session_state.tracker.update(final_state)
            summary = st.session_state.tracker.get_summary()

            # Quản lý số lần cảnh báo
            is_warning = ("WARNING" in final_state)
            if is_warning and not st.session_state.prev_warning_state:
                st.session_state.warning_count += 1
            st.session_state.prev_warning_state = is_warning

            # 6. Đồng bộ toàn bộ dữ liệu vào Data Manager
            update_data({
                "focus_score": focus_score,
                "state": final_state,
                "warning_count": st.session_state.warning_count,
                "study_time": summary["study_time"],
                "focused_time": summary["focused_time"],
                "distracted_time": summary["distracted_time"],
                "sleepy_time": summary["sleepy_time"],
                "absent_time": summary["absent_time"],
                "ear": ear,
                "blink_count": blink_count,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll
            })

            # 7. Render Frame ảnh Camera liên tục
            frame_placeholder.image(
                annotated_frame,
                channels="BGR",
                width=camera_width
            )

            # 8. Render Dashboard Realtime (Điều tiết cập nhật mỗi 0.25s để chống quá tải WebSocket / RAM)
            now = time()
            if (now - last_ui_update_time >= 0.25) or (final_state != last_ui_state):
                last_ui_update_time = now
                last_ui_state = final_state
                data = get_data()
                with realtime_placeholder:
                    show_realtime(data)

            # Nhường một nhịp ngắn (8ms) để giảm tải CPU và giữ luồng WebSocket ổn định
            sleep(0.008)

    finally:
        camera.release()


# =========================
# DASHBOARD ANALYTICS & REPORTS (TABS)
# =========================

st.divider()

st.subheader("📊 Tổng Kết & Phân Tích Phiên Học")

data = get_data()

tab_report, tab_chart, tab_stats = st.tabs([
    "📋 Báo Cáo & Khuyến Nghị",
    "📈 Biểu Đồ Thời Gian",
    "⏱️ Thống Kê Chi Tiết"
])

with tab_report:
    show_report(data)
    show_recommendations(data)

with tab_chart:
    show_analytics(data)

with tab_stats:
    show_statistics(data)