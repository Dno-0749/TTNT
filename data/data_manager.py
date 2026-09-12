from data.sample_data import data


REQUIRED_FIELDS = [
    "focus_score",
    "state",
    "warning_count",
    "study_time",
    "focused_time",
    "distracted_time",
    "sleepy_time",
    "absent_time",
    "focus_ratio",
    "ear",
    "eye_state",
    "closure_duration",
    "blink_count",
    "yaw",
    "pitch",
    "roll",
    "head_direction_vi",
    "ml_state"
]


def get_data():
    return data.copy()


def update_data(new_data):
    for key in REQUIRED_FIELDS:
        if key in new_data:
            data[key] = new_data[key]

    return data.copy()


def reset_session_data():
    data["focus_score"] = 100
    data["state"] = "READY"
    data["warning_count"] = 0
    data["study_time"] = 0.0
    data["focused_time"] = 0.0
    data["distracted_time"] = 0.0
    data["sleepy_time"] = 0.0
    data["absent_time"] = 0.0
    data["focus_ratio"] = 100.0
    data["ear"] = 0.32
    data["eye_state"] = "OPEN"
    data["closure_duration"] = 0.0
    data["blink_count"] = 0
    data["yaw"] = 0.0
    data["pitch"] = 0.0
    data["roll"] = 0.0
    data["head_direction_vi"] = "NHÌN THẲNG"
    data["ml_state"] = "Focused"
    return data.copy()