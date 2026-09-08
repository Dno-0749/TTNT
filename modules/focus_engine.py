import os
import joblib
import pandas as pd
from time import time

class FocusEngine:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "focus_model.pkl")
        
        self.model = None
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
            except Exception:
                pass
        
        # Biến Temporal Analysis
        self.distracted_start_time = None
        self.current_state = "FOCUSED"

    def predict_state(self, face_count, ear, blink_count, closure_duration, yaw, pitch=0.0, roll=0.0):
        if self.model is None:
            return "UNKNOWN"
        features = pd.DataFrame([{
            "face_count": face_count, "ear": ear, "blink_count": blink_count,
            "closure_duration": closure_duration, "yaw": yaw, "pitch": pitch, "roll": roll
        }])
        try:
            return self.model.predict(features)[0]
        except Exception:
            return "UNKNOWN"

    def calculate_score(self, eye_data, head_yaw, face_count, ml_state):
        now = time()
        
        # Phân tích theo thời gian (Temporal Analysis)
        is_raw_distracted = (abs(head_yaw) > 25.0) or (ml_state in ["Distracted", "Sleepy", "Absent"])

        if is_raw_distracted:
            if self.distracted_start_time is None:
                self.distracted_start_time = now
            
            elapsed = now - self.distracted_start_time
            if elapsed < 1.5:
                self.current_state = "FOCUSED" # Dưới 1.5s chưa phạt
            elif 1.5 <= elapsed < 4.0:
                self.current_state = "DISTRACTED"
            else:
                self.current_state = "WARNING" # Trên 4s cảnh báo mạnh
        else:
            self.distracted_start_time = None
            self.current_state = "FOCUSED"

        # Tính Focus Score
        if face_count == 0:
            return 0.0, "Absent"

        eye_score = 100.0 if eye_data.get("eye_state") == "OPEN" else 0.0
        head_score = max(0.0, 100.0 - abs(head_yaw) * 2.0)
        face_score = 100.0 if face_count == 1 else 0.0

        score = (head_score * 0.4) + (eye_score * 0.3) + (face_score * 0.3)

        return round(score, 1), self.current_state