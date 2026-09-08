import streamlit as st


def calculate_realtime(feature_dict, warning_count=0):
    """
    Chuyển dữ liệu Head Pose thật từ Camera thành trạng thái học tập.
    """
    face_detected = feature_dict.get("face_detected", False)
    is_distracted = feature_dict.get("is_distracted_pose", True)
    head_direction_vi = feature_dict.get("head_direction_vi", "KHÔNG THẤY KHUÔN MẶT")

    # 1. Trường hợp không thấy khuôn mặt
    if not face_detected:
        focus_score = 40
        state = "NO FACE DETECTED"
        warning = True
        warning_count += 1

    # 2. Trường hợp quay đầu (Mất tập trung: Trái, Phải, Ngẩng, Cúi)
    elif is_distracted:
        focus_score = 60
        state = f"DISTRACTED ({head_direction_vi})"
        warning = True
        warning_count += 1

    # 3. Trường hợp nhìn thẳng (Tập trung)
    else:
        focus_score = 95
        state = "FOCUSED"
        warning = False

    return {
        "focus_score": focus_score,
        "state": state,
        "warning": warning,
        "warning_count": warning_count,
        "yaw": feature_dict.get("yaw", 0.0),
        "pitch": feature_dict.get("pitch", 0.0),
        "roll": feature_dict.get("roll", 0.0)
    }


def show_realtime(data):
    st.subheader("📊 Realtime Monitoring")

    # Hiển thị 3 chỉ số chính
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Focus Score",
            f"{data['focus_score']} / 100"
        )

    with col2:
        st.metric(
            "Current State",
            data["state"]
        )

    with col3:
        st.metric(
            "Warnings",
            data["warning_count"]
        )

    # Hiển thị chi tiết góc quay đầu từ Head Pose
    if "yaw" in data:
        st.caption(f"📐 Góc Yaw: **{data['yaw']}°** | Pitch: **{data['pitch']}°** | Roll: **{data['roll']}°**")


def show_statistics(data):
    st.divider()

    st.subheader("⏱️ Session Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Study Time",
            f"{data['study_time']} min"
        )

    with col2:
        st.metric(
            "Focused Time",
            f"{data['focused_time']} min"
        )

    with col3:
        st.metric(
            "Distracted Time",
            f"{data['distracted_time']} min"
        )

    with col4:
        st.metric(
            "Sleepy Time",
            f"{data['sleepy_time']} min"
        )