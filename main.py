import os
from time import monotonic

import cv2
import mediapipe as mp

from modules.camera import Camera
from modules.eye_analyzer import EyeAnalyzer
from modules.face_detector import FaceDetector


def main():
    camera = None
    landmarker = None
    try:
        camera = Camera(
            camera_index=0,
            width=640,
            height=480
        )

        print("Camera đã được khởi động.")
        face_detector = FaceDetector()
        eye_analyzer = EyeAnalyzer()
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models",
            "face_landmarker.task",
        )
        if not os.path.exists(model_path):
            raise RuntimeError(f"Không tìm thấy model: {model_path}")

        landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            landmarker_options
        )
        print("Face Detector đã được khởi động.")
        print("Eye Analyzer đã được khởi động.")
        print("Nhấn Q để thoát.")
        print("-" * 50)
        while True:
            frame = camera.get_frame()
            if frame is None:
                print("Không thể đọc frame từ camera.")
                break
            faces = face_detector.detect(frame)
            face_count = face_detector.count_faces(faces)
            face_data = face_detector.get_face_data(faces)

            frame = face_detector.draw_faces(
                frame,
                faces
            )
            eye_result = None
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect(mp_image)
            if detection_result.face_landmarks:
                eye_result = eye_analyzer.process_landmarks(
                    detection_result.face_landmarks[0],
                    timestamp=monotonic(),
                )

            cv2.putText(
                frame,
                f"Faces: {face_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            if face_count == 0:

                status = "No face detected"

            elif face_count == 1:

                status = "1 face detected"

            else:

                status = f"{face_count} faces detected"

            cv2.putText(
                frame,
                status,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            if eye_result is not None:
                sleepy_label = "SLEEPY" if eye_result["is_sleepy"] else "ALERT"
                sleepy_color = (0, 0, 255) if eye_result["is_sleepy"] else (0, 255, 0)
                cv2.putText(
                    frame,
                    f"EAR: {eye_result['average_ear']:.2f} | Eyes: {eye_result['eye_state']}",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Blinks: {eye_result['blink_count']} | Closure: {eye_result['closure_duration']:.1f}s",
                    (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Status: {sleepy_label}",
                    (20, 205),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    sleepy_color,
                    2,
                )
            cv2.putText(
                frame,
                "Press Q to quit",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            cv2.imshow(
                "Camera + Face Detection",
                frame
            )
            print(
                f"Face count: {face_data['face_count']} | "
                f"Faces: {face_data['faces']}"
            )
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Đang thoát chương trình...")
                break
    except RuntimeError as error:
        print(f"Lỗi: {error}")
    except KeyboardInterrupt:
        print("\nChương trình bị dừng bởi người dùng.")

    finally:
        if landmarker is not None:
            landmarker.close()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        print("Camera đã được giải phóng.")
        print("Chương trình kết thúc.")


if __name__ == "__main__":
    main()