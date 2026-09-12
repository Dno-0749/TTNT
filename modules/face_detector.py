import cv2


class FaceDetector:
    def __init__(
        self,
        scale_factor=1.1,
        min_neighbors=5,
        min_size=(30, 30)
    ):
        """
        Khởi tạo Face Detector.
        Sử dụng Haar Cascade có sẵn trong OpenCV.
        """
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(
                "Không thể tải Haar Cascade Face Detector"
            )
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame):
        """
        Phát hiện tất cả khuôn mặt trong frame.

        Returns:
            Danh sách các bounding box:
            [
                (x, y, width, height),
                ...
            ]
        """
        if frame is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        return faces

    def count_faces(self, faces):
        """
        Đếm số lượng khuôn mặt.
        """
        return len(faces)

    def draw_faces(self, frame, faces):
        """
        Vẽ bounding box quanh từng khuôn mặt.
        """
        for index, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"Face {index + 1}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        return frame

    def draw_status(self, frame, faces):
        """
        Xử lý và hiển thị cảnh báo trực quan cho 2 trường hợp đặc biệt:
            - Không phát hiện khuôn mặt nào (face_count == 0)
            - Phát hiện nhiều hơn 1 khuôn mặt (face_count > 1)

        Trả về frame đã gắn cảnh báo (nếu có).
        """
        count = self.count_faces(faces)

        if count == 0:
            cv2.putText(
                frame,
                "Khong phat hien khuon mat",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        elif count > 1:
            cv2.putText(
                frame,
                f"Canh bao: phat hien {count} khuon mat",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2
            )

        return frame

    def get_face_data(self, faces):
        """
        Chuẩn hóa dữ liệu khuôn mặt.

        Output:
        {
            "face_count": 2,
            "status": "single_face" | "no_face" | "multiple_faces",
            "faces": [
                {"x": 100, "y": 80, "width": 120, "height": 120},
                ...
            ]
        }
        """
        face_data = []
        for (x, y, w, h) in faces:
            face_data.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            })

        count = len(face_data)
        if count == 0:
            status = "no_face"
        elif count == 1:
            status = "single_face"
        else:
            status = "multiple_faces"

        return {
            "face_count": count,
            "status": status,
            "faces": face_data
        }

    def detect_and_get_data(self, frame):
        """
        Hàm tiện ích:
        Frame -> Detect -> Chuẩn hóa dữ liệu

        Returns:
            face_data (dict)
        """
        faces = self.detect(frame)
        return self.get_face_data(faces)
if __name__ == "__main__":
    from modules.camera import Camera

    detector = FaceDetector()

    with Camera(camera_index=0) as cam:
        print("Nhấn 'q' để thoát...")
        while True:
            frame = cam.get_frame()
            if frame is None:
                print("Không đọc được frame, dừng chương trình.")
                break

            faces = detector.detect(frame)
            frame = detector.draw_faces(frame, faces)
            frame = detector.draw_status(frame, faces)

            cv2.imshow("Face Detection Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()