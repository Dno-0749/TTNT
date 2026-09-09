from time import time

class SessionTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time()
        self.total_focused_time = 0.0
        self.total_distracted_time = 0.0
        self.total_sleepy_time = 0.0
        self.total_absent_time = 0.0
        self.last_update = time()

    def update(self, state):
        now = time()
        dt = now - self.last_update
        self.last_update = now

        state_str = str(state).upper()

        if "FOCUSED" in state_str:
            self.total_focused_time += dt
        elif "SLEEP" in state_str:
            self.total_sleepy_time += dt
        elif "ABSENT" in state_str or "NO FACE" in state_str:
            self.total_absent_time += dt
        elif "DISTRACTED" in state_str or "WARNING" in state_str:
            self.total_distracted_time += dt
        else:
            self.total_focused_time += dt

    def get_summary(self):
        total_time = max(0.1, time() - self.start_time)
        focus_ratio = round((self.total_focused_time / total_time) * 100, 1)
        return {
            "study_time": round(total_time / 60.0, 2),
            "focused_time": round(self.total_focused_time / 60.0, 2),
            "distracted_time": round(self.total_distracted_time / 60.0, 2),
            "sleepy_time": round(self.total_sleepy_time / 60.0, 2),
            "absent_time": round(self.total_absent_time / 60.0, 2),
            "total_duration_sec": int(total_time),
            "focused_sec": round(self.total_focused_time, 1),
            "distracted_sec": round(self.total_distracted_time, 1),
            "sleepy_sec": round(self.total_sleepy_time, 1),
            "absent_sec": round(self.total_absent_time, 1),
            "focus_ratio": min(100.0, max(0.0, focus_ratio))
        }