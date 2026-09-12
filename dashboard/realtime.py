import streamlit as st


def calculate_realtime(feature_dict, current_state="FOCUSED", warning_count=0):
    """
    Chuyển dữ liệu Head Pose và Eye từ Camera thành thông tin hiển thị thời gian thực.
    """
    face_detected = feature_dict.get("face_detected", False)
    is_distracted = feature_dict.get("is_distracted_pose", True)
    head_direction_vi = feature_dict.get("head_direction_vi", "KHÔNG THẤY KHUÔN MẶT")

    warning = ("WARNING" in current_state) or ("DISTRACTED" in current_state) or (not face_detected)

    return {
        "state": current_state,
        "warning": warning,
        "warning_count": warning_count,
        "yaw": feature_dict.get("yaw", 0.0),
        "pitch": feature_dict.get("pitch", 0.0),
        "roll": feature_dict.get("roll", 0.0),
        "head_direction_vi": head_direction_vi
    }


def show_realtime(data):
    score = int(data.get("focus_score", 100))
    state = data.get("state", "READY").upper()
    warning_count = int(data.get("warning_count", 0))
    ear = float(data.get("ear", 0.32))
    eye_state = data.get("eye_state", "OPEN").upper()
    closure_dur = float(data.get("closure_duration", 0.0))
    blinks = int(data.get("blink_count", 0))
    yaw = float(data.get("yaw", 0.0))
    pitch = float(data.get("pitch", 0.0))
    roll = float(data.get("roll", 0.0))
    head_dir = data.get("head_direction_vi", "NHÌN THẲNG").upper()
    focus_ratio = float(data.get("focus_ratio", 100.0))
    ml_state = data.get("ml_state", "Focused")

    # Xác định màu sắc, badge và mô tả rõ ràng, độ tương phản cao
    if "WARNING" in state:
        state_badge = '<span style="background-color: #dc2626; color: #ffffff; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 0.92rem; letter-spacing: 0.3px; box-shadow: 0 0 12px rgba(220,38,38,0.6);">🚨 CẢNH BÁO VI PHẠM</span>'
        score_color = "#f87171"
        score_desc = "Đang mất tập trung kéo dài (>4s) - Cần chấn chỉnh ngay!"
    elif "SLEEP" in state:
        state_badge = '<span style="background-color: #7c3aed; color: #ffffff; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 0.92rem; letter-spacing: 0.3px; box-shadow: 0 0 12px rgba(124,58,237,0.6);">😴 CẢNH BÁO BUỒN NGỦ</span>'
        score_color = "#c084fc"
        score_desc = "Mắt nhắm kéo dài hoặc tần suất chớp mắt bất thường"
    elif "DISTRACTED" in state:
        state_badge = '<span style="background-color: #d97706; color: #ffffff; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 0.92rem; letter-spacing: 0.3px; box-shadow: 0 0 12px rgba(217,119,6,0.6);">👀 ĐANG XAO NHÃNG</span>'
        score_color = "#fbbf24"
        score_desc = "Đầu quay khỏi màn hình học tập"
    elif "ABSENT" in state or "NO FACE" in state:
        state_badge = '<span style="background-color: #475569; color: #ffffff; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 0.92rem;">⚪ VẮNG MẶT</span>'
        score_color = "#94a3b8"
        score_desc = "Không tìm thấy khuôn mặt học sinh trong khung hình"
    else:
        state_badge = '<span style="background-color: #059669; color: #ffffff; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 0.92rem; letter-spacing: 0.3px; box-shadow: 0 0 12px rgba(5,150,105,0.5);">🟢 TẬP TRUNG TỐT</span>'
        score_color = "#34d399"
        score_desc = "Duy trì chú ý xuất sắc vào bài học"

    # ==========================================
    # CARD 1: HERO FOCUS SCORE & LIVE STATUS
    # ==========================================
    st.markdown(
        f"""
        <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 14px; padding: 18px 20px; margin-bottom: 14px; box-shadow: 0 4px 16px rgba(0,0,0,0.35);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 0.85rem; font-weight: 700; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.8px;">🎯 ĐÁNH GIÁ ĐỘ TẬP TRUNG HIỆN TẠI</span>
                {state_badge}
            </div>
            <div style="display: flex; align-items: baseline; justify-content: space-between; margin: 4px 0 10px 0;">
                <div>
                    <span style="font-size: 3.2rem; font-weight: 900; color: {score_color}; line-height: 1; letter-spacing: -1px;">{score}</span>
                    <span style="font-size: 1.25rem; color: #e2e8f0; font-weight: 700; margin-left: 4px;">/ 100 Điểm</span>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 600;">Tỷ lệ tập trung phiên:</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: #38bdf8;">{focus_ratio:.1f}%</div>
                </div>
            </div>
            <div style="font-size: 0.88rem; color: #f1f5f9; font-weight: 600; padding: 6px 12px; background-color: #0f172a; border-radius: 8px; border-left: 4px solid {score_color};">
                💬 {score_desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Thanh tiến trình đo điểm số
    st.progress(max(0, min(100, score)) / 100.0)

    st.write("")

    # ==========================================
    # CARD 2: PHÂN TÍCH MẮT & ĐỘ MỎI (EYE BIOMETRICS)
    # ==========================================
    is_eye_open = (eye_state == "OPEN") or (ear >= 0.20)
    eye_badge_color = "#059669" if is_eye_open else "#dc2626"
    eye_badge_text = "ĐANG MỞ" if is_eye_open else "ĐANG NHẮM"

    fatigue_level = "Tỉnh táo" if closure_dur < 1.0 else ("Nguy cơ mỏi" if closure_dur < 2.0 else "Ngủ gật!")
    fatigue_color = "#34d399" if closure_dur < 1.0 else ("#fbbf24" if closure_dur < 2.0 else "#f87171")

    st.markdown(
        f"""
        <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 14px; padding: 16px 18px; margin-bottom: 14px; box-shadow: 0 4px 14px rgba(0,0,0,0.25);">
            <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 12px; display: flex; align-items: center; gap: 6px;">
                👁️ PHÂN TÍCH THỊ GIÁC & ĐỘ MỎI MẮT
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div style="background-color: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 10px 12px;">
                    <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 600;">Độ mở mắt (EAR)</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #ffffff; margin: 2px 0;">{ear:.2f}</div>
                    <div style="font-size: 0.75rem; color: #cbd5e1;">Chuẩn bình thường: &ge; 0.21</div>
                </div>
                <div style="background-color: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 10px 12px;">
                    <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 600;">Trạng thái mắt</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: {eye_badge_color}; margin: 5px 0;">
                        <span style="background-color: {eye_badge_color}; color: white; padding: 3px 10px; border-radius: 6px; font-size: 0.85rem;">{eye_badge_text}</span>
                    </div>
                    <div style="font-size: 0.75rem; color: #cbd5e1;">Nhắm liên tục: <b>{closure_dur:.1f}s</b></div>
                </div>
                <div style="background-color: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 10px 12px;">
                    <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 600;">Tổng số lần chớp mắt</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #f8fafc; margin: 2px 0;">{blinks} <span style="font-size: 0.85rem; font-weight: 500; color: #94a3b8;">lần</span></div>
                    <div style="font-size: 0.75rem; color: #cbd5e1;">Tần suất sinh lý tự nhiên</div>
                </div>
                <div style="background-color: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 10px 12px;">
                    <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 600;">Mức độ mỏi mắt</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: {fatigue_color}; margin: 3px 0;">{fatigue_level}</div>
                    <div style="font-size: 0.75rem; color: #cbd5e1;">Ngưỡng ngủ gật: &gt; 2.0s</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================
    # CARD 3: TƯ THẾ ĐẦU & KHÔNG GIAN 3D (HEAD POSE)
    # ==========================================
    # Đánh giá góc Yaw (Xoay) và Pitch (Ngẩng/Cúi)
    yaw_status = "Cân bằng" if abs(yaw) <= 15 else ("Quay Trái" if yaw < -15 else "Quay Phải")
    pitch_status = "Vừa tầm" if abs(pitch) <= 15 else ("Ngẩng đầu" if pitch > 15 else "Cúi đầu")
    yaw_color = "#34d399" if abs(yaw) <= 15 else "#fbbf24"
    pitch_color = "#34d399" if abs(pitch) <= 15 else "#fbbf24"

    st.markdown(
        f"""
        <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 14px; padding: 16px 18px; margin-bottom: 14px; box-shadow: 0 4px 14px rgba(0,0,0,0.25);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.8px;">🗣️ TƯ THẾ ĐẦU & HƯỚNG NHÌN 3D</span>
                <span style="background-color: #0f172a; color: #ffffff; border: 1px solid #38bdf8; padding: 3px 10px; border-radius: 8px; font-weight: 700; font-size: 0.82rem;">{head_dir}</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; text-align: center;">
                <div style="background-color: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 10px 6px;">
                    <div style="font-size: 0.74rem; color: #94a3b8; font-weight: 600;">Góc Yaw (Ngang)</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: {yaw_color}; margin: 2px 0;">{yaw:+.1f}°</div>
                    <div style="font-size: 0.72rem; color: #cbd5e1;">{yaw_status}</div>
                </div>
                <div style="background-color: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 10px 6px;">
                    <div style="font-size: 0.74rem; color: #94a3b8; font-weight: 600;">Góc Pitch (Dọc)</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: {pitch_color}; margin: 2px 0;">{pitch:+.1f}°</div>
                    <div style="font-size: 0.72rem; color: #cbd5e1;">{pitch_status}</div>
                </div>
                <div style="background-color: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 10px 6px;">
                    <div style="font-size: 0.74rem; color: #94a3b8; font-weight: 600;">Góc Roll (Nghiêng)</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: #34d399; margin: 2px 0;">{roll:+.1f}°</div>
                    <div style="font-size: 0.72rem; color: #cbd5e1;">Cân bằng</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================
    # CARD 4: TIẾN TRÌNH BUỔI HỌC & CẢNH BÁO
    # ==========================================
    warn_box_color = "#ef4444" if warning_count > 3 else ("#f59e0b" if warning_count > 0 else "#10b981")
    study_sec = int(round(float(data.get("study_time", 0.0)) * 60))
    m = study_sec // 60
    s = study_sec % 60
    time_str = f"{m}m {s}s" if m > 0 else f"{s}s"

    st.markdown(
        f"""
        <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 14px; padding: 14px 18px; box-shadow: 0 4px 14px rgba(0,0,0,0.25);">
            <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px;">
                ⏱️ GIÁM SÁT PHIÊN HỌC & AI MODEL
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; background-color: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;">
                <span style="font-size: 0.88rem; color: #cbd5e1; font-weight: 600;">⏱️ Thời gian học phiên hiện tại:</span>
                <span style="font-size: 1.15rem; font-weight: 800; color: #ffffff;">{time_str}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; background-color: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 10px 14px;">
                <span style="font-size: 0.88rem; color: #cbd5e1; font-weight: 600;">🚨 Tổng số lần cảnh báo phạt:</span>
                <span style="font-size: 1.15rem; font-weight: 800; color: {warn_box_color};">{warning_count} lần</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px; font-size: 0.78rem; color: #94a3b8;">
                <span>🤖 Dự đoán ML: <b style="color:#ffffff;">{ml_state}</b></span>
                <span style="color:#34d399;">● AI Engine Live</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_statistics(data):
    def format_time(minutes):
        if minutes is None or minutes <= 0:
            return "0s"
        total_sec = int(round(minutes * 60))
        if total_sec < 60:
            return f"{total_sec}s"
        m = total_sec // 60
        s = total_sec % 60
        return f"{m}m {s}s" if s > 0 else f"{m}m"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "⏱️ Tổng thời gian học",
            format_time(data.get("study_time", 0.0))
        )

    with col2:
        st.metric(
            "🎯 Thời gian tập trung",
            format_time(data.get("focused_time", 0.0)),
            delta="Hiệu quả" if data.get("focused_time", 0) > 0 else None
        )

    with col3:
        st.metric(
            "👀 Thời gian xao nhãng",
            format_time(data.get("distracted_time", 0.0)),
            delta_color="inverse"
        )

    with col4:
        st.metric(
            "😴 Thời gian buồn ngủ",
            format_time(data.get("sleepy_time", 0.0)),
            delta_color="inverse"
        )