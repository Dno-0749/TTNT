from data.sample_data import data


REQUIRED_FIELDS = [
    "focus_score",
    "state",
    "warning_count",
    "study_time",
    "focused_time",
    "distracted_time",
    "sleepy_time"
]


def get_data():
    return data.copy()


def update_data(new_data):
    for key in REQUIRED_FIELDS:
        if key in new_data:
            data[key] = new_data[key]

    return data.copy()