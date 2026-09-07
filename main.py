import cv2
from camera import Camera
from face_detector import FaceDetector
def main():
    camera = None
    try:
        camera = Camera(
            camera_index=0,
            width=640,
            height=480
        )

        print("Camera đã được khởi động.")
        face_detector = FaceDetector()
        print("Face Detector đã được khởi động.")
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
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        print("Camera đã được giải phóng.")
        print("Chương trình kết thúc.")


if __name__ == "__main__":
    main()