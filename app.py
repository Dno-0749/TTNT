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
from modules.face_detector import FaceDetector
from modules.face_tracker import FaceTracker
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

if "face_tracker" not in st.session_state:
    st.session_state.face_tracker = FaceTracker()


def stop_camera_observation():
    st.session_state.camera_toggle = False
    st.session_state.face_tracker.reset()
    
    # Hiệu ứng khen thưởng nếu học tốt
    try:
        data = get_data()
        if data and data.get("focus_score", 0) >= 80 and data.get("study_time", 0) >= 10:
            st.balloons()
    except Exception:
        pass


def reset_study_session():
    st.session_state.tracker.reset()
    st.session_state.focus_engine.reset()
    st.session_state.eye_analyzer.reset()
    st.session_state.face_tracker.reset()
    st.session_state.warning_count = 0
    st.session_state.prev_warning_state = False
    reset_session_data()


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
        step=40,
        help="Độ phân giải càng cao, AI nhận diện càng chính xác nhưng sẽ tiêu tốn nhiều tài nguyên CPU của máy hơn."
    )
    camera_height = int(camera_width * 480 / 640)
    st.caption(f"Độ phân giải: {camera_width} × {camera_height}")

    st.divider()
    st.caption("🤖 Model: Machine Learning Classification")
    st.caption("👁️ Eye Tracking: MediaPipe Face Mesh (EAR)")
    st.caption("🗣️ Head Pose: 3D SolvePnP (Yaw/Pitch/Roll)")


# =========================
# CUSTOM CSS STYLING
# =========================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
    }
    
    /* Background override for Streamlit app */
    .stApp {
        background-color: #F9FBF9;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F4F7F4 100%);
        border-right: 1px solid #E6EBE6;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 2.2rem 1.35rem 1.5rem;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #3D405B;
        letter-spacing: -0.01em;
        font-weight: 700;
    }
    [data-testid="stSidebar"] label {
        color: #5B6370;
        font-weight: 600;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #8C99A6;
        font-size: 0.8rem;
        line-height: 1.5;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        min-height: 2.8rem;
        border: 1px solid #E0E6E0;
        border-radius: 14px;
        background: #FFFFFF;
        box-shadow: 0 4px 15px rgba(132, 165, 157, 0.08);
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] {
        padding: 0.2rem 0 0.65rem;
    }
    [data-testid="stSidebar"] hr {
        margin: 1.35rem 0;
        border-color: #E6EBE6;
    }
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 14px;
        border: 1px solid #D6DFD6;
        font-weight: 700;
        background-color: #FFFFFF;
        color: #3D405B;
        transition: all 0.3s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: #84A59D;
        color: #84A59D;
        background-color: #F4F7F4;
    }
    div[data-testid="stImage"] img {
        border-radius: 18px;
        border: 4px solid #FFFFFF;
        box-shadow: 0 8px 30px rgba(132, 165, 157, 0.15);
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E6EBE6;
        padding: 16px 20px;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(132, 165, 157, 0.06);
    }
    div[data-testid="stMetric"] label {
        color: #7D8C9B !important;
        font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #3D405B !important;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 10px 22px;
        font-weight: 600;
        color: #5B6370;
        background: #F4F7F4;
        border: none;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #84A59D;
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(132, 165, 157, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# =========================
# CAMERA CONTROLS
# =========================

st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
        <span style="font-size: 1.25rem; font-weight: 700; color: #3D405B; font-family: 'Nunito', sans-serif;">📷 Luồng Giám Sát Camera</span>
    </div>
    """,
    unsafe_allow_html=True
)

ctrl_col1, ctrl_col2 = st.columns([1.2, 1])
with ctrl_col1:
    start_camera = st.toggle("🎥 Kích hoạt Camera Giám sát", value=False, key="camera_toggle")
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        st.button(
            "⏹️ Kết thúc quan sát",
            use_container_width=True,
            disabled=not start_camera,
            on_click=stop_camera_observation,
        )
    with action_col2:
        st.button(
            "🔄 Làm mới phiên học",
            use_container_width=True,
            on_click=reset_study_session,
        )
with ctrl_col2:
    with st.popover("⚙️ Cài đặt Camera"):
        camera_idx = st.selectbox("Cổng Camera (Index)", options=[0, 1, 2], index=0, help="Thay đổi nếu bạn dùng nhiều Webcam (0 là mặc định).")
        camera_width = st.slider("Chiều rộng hiển thị (px)", 360, 960, 640, 40, help="Tăng để xem rõ hơn, giảm để ứng dụng chạy nhẹ hơn.")
        camera_height = int(camera_width * 480 / 640)
        st.caption(f"Độ phân giải: {camera_width} × {camera_height}")

st.divider()

# =========================
# MAIN DASHBOARD LAYOUT (2 COLUMNS)
# =========================

col_cam, col_hud = st.columns([1, 1], gap="large")

with col_cam:
    warning_placeholder = st.empty()
    frame_placeholder = st.empty()

with col_hud:
    st.markdown(
        """
        <div style="margin-bottom: 8px;">
            <span style="font-size: 1.25rem; font-weight: 700; color: #3D405B; font-family: 'Nunito', sans-serif;">📊 Chỉ Số Thời Gian Thực</span>
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
            <div style="background: #FFFFFF; border: 2px dashed #A3B18A; border-radius: 16px; min-height: 430px; padding: 30px 20px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; margin: 4px 0 14px 0; box-sizing: border-box; box-shadow: 0 4px 20px rgba(132, 165, 157, 0.05); font-family: 'Nunito', sans-serif;">
                <div style="font-size: 3.5rem; margin-bottom: 8px;">📹</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #3D405B;">Camera Đang Ở Trạng Thái Chờ</div>
                <div style="font-size: 0.88rem; color: #7D8C9B; max-width: 400px; margin: 8px auto 0 auto; line-height: 1.5;">
                    Gạt nút <b>"Kích hoạt Camera Giám sát"</b> bên dưới để AI tự động phân tích hướng nhìn và mức độ tập trung.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with realtime_placeholder:
        show_realtime(get_data())

else:
    with st.spinner("🤖 Đang khởi động AI Models và kết nối Camera..."):
        camera = Camera(
            camera_index=camera_idx,
            width=camera_width,
            height=camera_height
        )
        estimator = HeadPoseEstimator()
        face_detector = FaceDetector()

    last_ui_update_time = 0.0
    last_ui_state = ""
    last_frame_update_time = 0.0
    frame_number = 0
    tracked_faces = []
    face_detection_interval = 3

    try:
        while start_camera:
            frame = camera.get_frame()
            frame_number += 1

            if frame is None:
                st.error("❌ Không thể đọc luồng dữ liệu từ camera. Vui lòng kiểm tra lại webcam.")
                break

            if frame_number % face_detection_interval == 0 or not tracked_faces:
                detected_faces = face_detector.detect(frame)
                tracked_faces = st.session_state.face_tracker.update(
                    face_detector.get_face_data(detected_faces)["faces"]
                )
            face_count = sum(face["visible"] for face in tracked_faces)

            annotated_frame, feature_dict = estimator.process_frame(frame)
            raw_landmarks = feature_dict.get("raw_landmarks")
            face_detected = feature_dict.get("face_detected", False)
            feature_dict["face_count"] = face_count
            face_detected = face_count > 0

            for face in tracked_faces:
                x = face["x"]
                y = face["y"]
                width = face["width"]
                height = face["height"]
                cv2.rectangle(
                    annotated_frame,
                    (x, y),
                    (x + width, y + height),
                    (157, 165, 132), # Sage Green BGR
                    2,
                )
                cv2.putText(
                    annotated_frame,
                    f"ID {face['id']}",
                    (x, max(22, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (95, 122, 224), # Terracotta BGR
                    2,
                    cv2.LINE_AA,
                )

            if face_count == 0:
                cv2.putText(
                    annotated_frame,
                    "NO FACE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (95, 122, 224),
                    2,
                    cv2.LINE_AA,
                )
            elif face_count > 1:
                cv2.putText(
                    annotated_frame,
                    f"WARNING: {face_count} FACES",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (95, 122, 224),
                    2,
                    cv2.LINE_AA,
                )

            # 4. Phân tích trạng thái mắt qua EyeAnalyzer (EAR, chớp mắt, buồn ngủ)
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
                eye_color = (138, 177, 163) if eye_metrics["eye_state"] == "OPEN" else (95, 122, 224)
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
                        "WARNING: SLEEPY / EYES CLOSED TOO LONG",
                        (20, camera_height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (95, 122, 224),
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
                st.toast("Cảnh báo: Bạn đang mất tập trung hoặc nhắm mắt quá lâu!", icon="⚠️")
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
            now = time()
            if now - last_frame_update_time >= (1 / 60):
                last_frame_update_time = now
                frame_placeholder.image(
                    annotated_frame,
                    channels="BGR",
                    width="stretch"
                )

            # 8. Render Dashboard Realtime (Điều tiết cập nhật mỗi 0.25s để chống quá tải WebSocket / RAM)
            if (now - last_ui_update_time >= 0.25) or (final_state != last_ui_state):
                last_ui_update_time = now
                last_ui_state = final_state
                data = get_data()
                with realtime_placeholder:
                    show_realtime(data)
                
                # Hiển thị Banner cảnh báo trực quan cỡ lớn ngay trên Camera
                if "SLEEPY" in final_state or eye_metrics.get("is_sleepy", False):
                    warning_placeholder.markdown(
                        """
                        <div style="background-color: #E07A5F; color: white; padding: 14px; border-radius: 14px; text-align: center; font-weight: 700; font-size: 1.15rem; margin-bottom: 12px; box-shadow: 0 4px 15px rgba(224, 122, 95, 0.4); font-family: 'Nunito', sans-serif;">
                            🚨 CẢNH BÁO: BẠN ĐANG NGỦ GỤC! HÃY MỞ MẮT RA!
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif "WARNING" in final_state or "DISTRACTED" in final_state:
                    warning_placeholder.markdown(
                        """
                        <div style="background-color: #F6BD60; color: #3D405B; padding: 12px; border-radius: 14px; text-align: center; font-weight: 700; font-size: 1.05rem; margin-bottom: 12px; box-shadow: 0 4px 15px rgba(246, 189, 96, 0.25); font-family: 'Nunito', sans-serif;">
                            ⚠️ NHẮC NHỞ: Vui lòng chú ý vào màn hình!
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    warning_placeholder.empty()

            # Nhường một nhịp ngắn (8ms) để giảm tải CPU và giữ luồng WebSocket ổn định
            sleep(0.004)

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