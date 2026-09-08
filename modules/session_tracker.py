from time import time

class SessionTracker:
    def __init__(self):
        self.start_time = time()
        self.total_focused_time = 0.0
        self.total_distracted_time = 0.0
        self.last_update = time()

    def update(self, state):
        now = time()
        dt = now - self.last_update
        self.last_update = now

        if state == "Focused":
            self.total_focused_time += dt
        elif state in ["Distracted", "Sleepy", "Absent"]:
            self.total_distracted_time += dt

    def get_summary(self):
        total_time = max(1.0, time() - self.start_time)
        focus_ratio = round((self.total_focused_time / total_time) * 100, 1)
        return {
            "total_duration_sec": int(total_time),
            "focused_sec": int(self.total_focused_time),
            "distracted_sec": int(self.total_distracted_time),
            "focus_ratio": focus_ratio
        }