import cv2
import os


class FaceDetector:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        cascade_path_1 = os.path.join(base_dir, "models", "haarcascade_frontalface_default.xml")
        cascade_path_2 = os.path.join(base_dir, "modules", "models", "haarcascade_frontalface_default.xml")

        if os.path.exists(cascade_path_1):
            cascade_path = cascade_path_1
        elif os.path.exists(cascade_path_2):
            cascade_path = cascade_path_2
        else:
            raise FileNotFoundError(f"Không tìm thấy file xml tại {cascade_path_1} hoặc {cascade_path_2}")

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces

    def count_faces(self, faces):
        return len(faces)

    def draw_faces(self, frame, faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame

    def draw_status(self, frame, faces):
        """
        Hiển thị trạng thái khuôn mặt.
        """
        face_count = len(faces)

        if face_count == 0:
            status = "NO FACE"
        elif face_count == 1:
            status = "SINGLE FACE"
        else:
            status = "MULTIPLE FACES"

        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        return frame

    def get_face_data(self, faces):
        """
        Trả về dữ liệu khuôn mặt.
        """
        count = len(faces)

        if count == 0:
            status = "no_face"
        elif count == 1:
            status = "single_face"
        else:
            status = "multiple_faces"

        face_list = []

        for (x, y, w, h) in faces:
            face_list.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            })

        return {
            "face_count": count,
            "status": status,
            "faces": face_list
        }

    def detect_and_get_data(self, frame):
        """
        Phát hiện khuôn mặt và trả về dữ liệu.
        """
        faces = self.detect(frame)

        return self.get_face_data(faces)


# ==============================
# TEST FACE DETECTOR
# ==============================

if __name__ == "__main__":

    print("=" * 50)
    print("SMARTSTUDY AI - FACE DETECTOR TEST")
    print("=" * 50)

    print("OpenCV version:", cv2.__version__)

    try:
        detector = FaceDetector()

        print("Haar Cascade: OK")
        print("Face Detector: OK")
        print()

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            print("Không thể mở camera.")
            exit()

        print("Camera: OK")
        print("Đang chạy camera...")
        print("Nhấn Q để thoát.")

        while True:

            ret, frame = camera.read()

            if not ret:
                print("Không đọc được frame từ camera.")
                break

            faces = detector.detect(frame)

            frame = detector.draw_faces(frame, faces)
            frame = detector.draw_status(frame, faces)

            cv2.imshow(
                "SMARTSTUDY AI - Face Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        camera.release()
        cv2.destroyAllWindows()

    except Exception as e:

        print()
        print("=" * 50)
        print("LOI:")
        print(str(e))
        print("=" * 50)