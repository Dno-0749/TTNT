import streamlit as st
import pandas as pd


def show_analytics(data):
    st.divider()

    st.subheader("📈 Study Analytics")

    study_time = data.get("study_time", 0.0)
    use_seconds = study_time < 3.0
    unit = "giây" if use_seconds else "phút"
    multiplier = 60.0 if use_seconds else 1.0

    chart_data = pd.DataFrame(
        {
            f"Thời gian ({unit})": [
                round(data.get("focused_time", 0.0) * multiplier, 1),
                round(data.get("distracted_time", 0.0) * multiplier, 1),
                round(data.get("sleepy_time", 0.0) * multiplier, 1),
                round(data.get("absent_time", 0.0) * multiplier, 1)
            ]
        },
        index=[
            "Tập trung (Focused)",
            "Xao nhãng (Distracted)",
            "Buồn ngủ (Sleepy)",
            "Vắng mặt (Absent)"
        ]
    )

    st.bar_chart(chart_data)


def show_report(data):
    st.divider()

    st.subheader("📋 Study Report")

    study_time = data.get("study_time", 0.0)
    focused_time = data.get("focused_time", 0.0)
    focus_rate = (focused_time / study_time * 100) if study_time > 0 else 0.0

    st.write(f"### Tỷ lệ tập trung (Focus Rate): **{focus_rate:.1f}%**")

    if study_time == 0:
        st.info("Chưa có dữ liệu phiên học. Hãy bật camera để bắt đầu theo dõi.")
    elif focus_rate >= 80:
        st.success("🌟 Xuất sắc! Bạn duy trì mức độ tập trung rất tốt trong buổi học.")
    elif focus_rate >= 60:
        st.warning("⚠️ Mức độ tập trung ở mức trung bình, cần giảm bớt xao nhãng.")
    else:
        st.error("🚨 Mức độ tập trung thấp! Bạn nên nghỉ giải lao ngắn và quay lại sau.")

    st.write(
        f"""
**Tổng kết phiên học:**
- Tổng thời gian học: **{data.get('study_time', 0.0):.2f} phút**
- Thời gian tập trung: **{data.get('focused_time', 0.0):.2f} phút**
- Thời gian xao nhãng: **{data.get('distracted_time', 0.0):.2f} phút**
- Thời gian buồn ngủ: **{data.get('sleepy_time', 0.0):.2f} phút**
- Số lần cảnh báo: **{data.get('warning_count', 0)} lần**
"""
    )


def show_recommendations(data):
    st.divider()

    st.subheader("💡 Recommendations")

    study_time = data.get("study_time", 0.0)
    if study_time == 0:
        st.caption("Khuyến nghị sẽ xuất hiện sau khi hệ thống ghi nhận thời gian học.")
        return

    focus_score = data.get("focus_score", 100)
    sleepy_time = data.get("sleepy_time", 0.0)
    distracted_time = data.get("distracted_time", 0.0)

    if focus_score >= 80:
        st.success("Thói quen học tập rất tốt! Hãy duy trì phong độ hiện tại.")
    elif focus_score >= 60:
        st.warning("Mức độ tập trung vừa phải. Hãy loại bỏ điện thoại và các tác nhân gây xao nhãng.")
    else:
        st.error("Bạn đang mất tập trung nghiêm trọng. Hãy áp dụng phương pháp Pomodoro 25/5.")

    if sleepy_time >= 0.5:
        st.info("😴 Hệ thống phát hiện dấu hiệu buồn ngủ. Hãy đứng dậy đi lại, uống nước hoặc rửa mặt.")

    if distracted_time >= 1.0:
        st.info("📵 Hãy tắt thông báo điện thoại và giữ mắt hướng vào bài giảng/màn hình học.")