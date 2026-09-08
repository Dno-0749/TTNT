import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="SMARTSTUDY AI",
    layout="wide"
)

# =========================
# TIÊU ĐỀ
# =========================

st.title("🎓 SMARTSTUDY AI")
st.write("AI Student Focus Monitoring System")

st.divider()


# =========================
# DỮ LIỆU MẪU TẠM THỜI
# =========================

data = {
    "focus_score": 87,
    "state": "FOCUSED",
    "warning_count": 2,

    "study_time": 120,
    "focused_time": 90,
    "distracted_time": 20,
    "sleepy_time": 10
}


# =========================
# REALTIME DASHBOARD
# =========================

st.subheader("📊 Realtime Monitoring")

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
# =========================
# SESSION STATISTICS
# =========================

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
# =========================
# ANALYTICS
# =========================

st.divider()

st.subheader("📈 Study Analytics")

chart_data = pd.DataFrame(
    {
        "Time": [
            data["focused_time"],
            data["distracted_time"],
            data["sleepy_time"]
        ]
    },
    index=[
        "Focused",
        "Distracted",
        "Sleepy"
    ]
)

st.bar_chart(chart_data)
# =========================
# STUDY REPORT
# =========================

st.divider()

st.subheader("📋 Study Report")

focus_rate = (
    data["focused_time"] / data["study_time"] * 100
)

st.write(f"### Focus Rate: {focus_rate:.1f}%")

if data["focus_score"] >= 80:
    st.success("Excellent! You are maintaining good focus.")
elif data["focus_score"] >= 60:
    st.warning("Your focus is acceptable but can be improved.")
else:
    st.error("Your focus level is low. Consider taking a short break.")

st.write(
    f"""
    **Session Summary**

    - Total Study Time: {data["study_time"]} minutes
    - Focused Time: {data["focused_time"]} minutes
    - Distracted Time: {data["distracted_time"]} minutes
    - Sleepy Time: {data["sleepy_time"]} minutes
    - Total Warnings: {data["warning_count"]}
    """
)
# =========================
# RECOMMENDATION
# =========================

st.divider()

st.subheader("💡 Recommendations")

if data["focus_score"] >= 80:
    st.success("Great job! Keep maintaining your current study habits.")

elif data["focus_score"] >= 60:
    st.warning(
        "Your focus is moderate. Try reducing distractions."
    )

else:
    st.error(
        "Your focus is low. Consider taking a short break."
    )


if data["sleepy_time"] >= 30:
    st.info(
        "You seem tired. Consider taking a 5-10 minute break."
    )

if data["distracted_time"] >= 30:
    st.info(
        "Try turning off notifications to reduce distractions."
    )