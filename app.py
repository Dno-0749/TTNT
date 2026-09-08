import os
import sys

# Thêm đường dẫn gốc để đảm bảo Python nhận diện đúng mọi module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import streamlit as st

from data.data_manager import get_data, update_data

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


# =========================
# CAMERA & REALTIME MONITORING
# =========================

st.subheader("📷 Camera Monitoring")

start_camera = st.checkbox("Start Camera")

if start_camera:

    # =========================
    # CAMERA SIZE
    # =========================

    camera_width = st.slider(
        "📐 Kích thước camera",
        min_value=320,
        max_value=1000,
        value=640,
        step=20
    )

    camera_height = int(camera_width * 480 / 640)

    st.caption(f"Kích thước hiển thị: {camera_width} × {camera_height}")

    # =========================
    # INIT OBJECTS
    # =========================

    camera = Camera(
        camera_index=0,
        width=camera_width,
        height=camera_height
    )

    estimator = HeadPoseEstimator()

    frame_placeholder = st.empty()
    realtime_placeholder = st.container()

    try:
        while start_camera:

            # 1. Lấy frame từ camera
            frame = camera.get_frame()

            if frame is None:
                st.error("❌ Không thể đọc luồng dữ liệu từ camera.")
                break

            # 2. Xử lý AI Head Pose & Các đặc trưng mặt/mắt
            annotated_frame, feature_dict = estimator.process_frame(frame)

            # Extract features an toàn
            face_count = feature_dict.get("face_count", 1)
            ear = feature_dict.get("ear", 0.3)
            blink_count = feature_dict.get("blink_count", 0)
            closure_duration = feature_dict.get("closure_duration", 0.0)
            yaw = feature_dict.get("yaw", 0.0)

            # 3. Dự đoán trạng thái bằng Model ML & Tính điểm Focus Engine
            ml_state = st.session_state.focus_engine.predict_state(
                face_count=face_count,
                ear=ear,
                blink_count=blink_count,
                closure_duration=closure_duration,
                yaw=yaw
            )

            focus_score, final_state = st.session_state.focus_engine.calculate_score(
                eye_data={"eye_state": "OPEN" if ear > 0.2 else "CLOSED", "is_sleepy": closure_duration > 1.5},
                head_yaw=yaw,
                face_count=face_count,
                ml_state=ml_state
            )

            # 4. Tính toán logic trạng thái Realtime
            realtime = calculate_realtime(
                feature_dict,
                warning_count=st.session_state.warning_count
            )

            # Ghi đè trạng thái & điểm số từ ML Model chuẩn
            realtime["focus_score"] = focus_score
            realtime["state"] = final_state

            # 5. Quản lý đếm Cảnh báo & Cập nhật Session Tracker
            if realtime["warning"] and not st.session_state.prev_warning_state:
                st.session_state.warning_count += 1
            st.session_state.prev_warning_state = realtime["warning"]

            st.session_state.tracker.update(final_state)

            # 6. Cập nhật dữ liệu hệ thống
            update_data({
                "focus_score": realtime["focus_score"],
                "state": realtime["state"],
                "warning_count": st.session_state.warning_count
            })

            # 7. Render Frame
            frame_placeholder.image(
                annotated_frame,
                channels="BGR",
                width=camera_width
            )

            # 8. Render Dashboard Realtime
            data = get_data()
            with realtime_placeholder:
                show_realtime(data)

    finally:
        camera.release()


# =========================
# DASHBOARD ANALYTICS & REPORTS
# =========================

st.divider()

st.subheader("📊 Session Analytics & Reports")

data = get_data()

show_statistics(data)

show_analytics(data)

show_report(data)

show_recommendations(data)