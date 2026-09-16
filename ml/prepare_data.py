import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from time import monotonic, time
import cv2
import mediapipe as mp

from modules.camera import Camera
from modules.eye_analyzer import EyeAnalyzer
from modules.face_detector import FaceDetector
def main():
    # 1. Xác định đường dẫn gốc dự án SMARTSTUDY_AI
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 2. Tạo thư mục data và đường dẫn dataset.csv
    dataset_dir = os.path.join(base_dir, "data")
    os.makedirs(dataset_dir, exist_ok=True)
    csv_path = os.path.join(dataset_dir, "dataset.csv")

    # 3. Kiểm tra file model MediaPipe
    model_path = os.path.join(base_dir, "models", "face_landmarker.task")
    if not os.path.exists(model_path):
        print(f"❌ LỖI: Không tìm thấy file model tại: {model_path}")
        return

    # 4. Tạo/Mở file CSV ghi dữ liệu
    file_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, mode="a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)

    if not file_exists:
        writer.writerow([
            "timestamp",
            "face_count",
            "ear",
            "blink_count",
            "closure_duration",
            "yaw",
            "pitch",
            "roll",
            "label"
        ])

    # 5. Khởi tạo các Module
    camera = Camera(camera_index=0, width=640, height=480)
    face_detector = FaceDetector()
    eye_analyzer = EyeAnalyzer()

    landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(landmarker_options)

    # 6. Đăng ký phím gán nhãn
    key_labels = {
        ord('1'): "Focused",
        ord('2'): "Distracted",
        ord('3'): "Sleepy",
        ord('4'): "Absent"
    }

    current_label = "Focused"
    is_recording = False

    print("=" * 60)
    print(" BỘ THU THẬP DỮ LIỆU CÂY NHÃN (DATA COLLECTOR)")
    print("=" * 60)
    print(" [1] : Focused | [2] : Distracted | [3] : Sleepy | [4] : Absent")
    print(" [R] : Bật/Tắt GHI DỮ LIỆU | [Q] : Thoát")
    print("=" * 60)

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                print("❌ Không đọc được luồng camera.")
                break

            faces = face_detector.detect(frame)
            face_count = face_detector.count_faces(faces)
            frame = face_detector.draw_faces(frame, faces)

            ear = 0.0
            blink_count = 0
            closure_duration = 0.0
            yaw, pitch, roll = 0.0, 0.0, 0.0

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect(mp_image)

            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                
                # Mắt
                eye_result = eye_analyzer.process_landmarks(landmarks, timestamp=monotonic())
                ear = eye_result["average_ear"]
                blink_count = eye_result["blink_count"]
                closure_duration = eye_result["closure_duration"]

                # Góc xoay Yaw xấp xỉ
                nose_bridge = landmarks[1]
                left_cheek = landmarks[234]
                right_cheek = landmarks[454]
                dist_left = abs(nose_bridge.x - left_cheek.x)
                dist_right = abs(nose_bridge.x - right_cheek.x)
                if dist_left + dist_right > 0:
                    yaw = ((dist_right - dist_left) / (dist_left + dist_right)) * 90.0

            # Ghi dòng dữ liệu vào CSV
            if is_recording:
                writer.writerow([
                    round(time(), 2),
                    face_count,
                    round(ear, 4),
                    blink_count,
                    round(closure_duration, 2),
                    round(yaw, 2),
                    round(pitch, 2),
                    round(roll, 2),
                    current_label
                ])

            # Hiển thị HUD trên giao diện OpenCV
            rec_status = "RECORDING..." if is_recording else "PAUSED"
            rec_color = (0, 0, 255) if is_recording else (128, 128, 128)

            cv2.putText(frame, f"REC: {rec_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rec_color, 2)
            cv2.putText(frame, f"LABEL: {current_label}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.2f} | YAW: {yaw:.1f} | Faces: {face_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Dataset Collector - SMARTSTUDY AI", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                is_recording = not is_recording
                print(f">> Trạng thái ghi dữ liệu: {is_recording}")
            elif key in key_labels:
                current_label = key_labels[key]
                print(f">> Chuyển nhãn ghi: {current_label}")

    finally:
        csv_file.close()
        landmarker.close()
        camera.release()
        cv2.destroyAllWindows()
        print(f"\n🎉 Dữ liệu đã lưu thành công tại: {csv_path}")


if __name__ == "__main__":
    main()