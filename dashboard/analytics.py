import streamlit as st
import pandas as pd


def show_analytics(data):
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


def show_report(data):
    st.divider()

    st.subheader("📋 Study Report")

    focus_rate = (
        data["focused_time"]
        / data["study_time"]
        * 100
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


def show_recommendations(data):
    st.divider()

    st.subheader("💡 Recommendations")

    if data["focus_score"] >= 80:
        st.success("Great job! Keep maintaining your current study habits.")
    elif data["focus_score"] >= 60:
        st.warning("Your focus is moderate. Try reducing distractions.")
    else:
        st.error("Your focus is low. Consider taking a short break.")

    if data["sleepy_time"] >= 30:
        st.info("You seem tired. Consider taking a 5-10 minute break.")

    if data["distracted_time"] >= 30:
        st.info("Try turning off notifications to reduce distractions.")