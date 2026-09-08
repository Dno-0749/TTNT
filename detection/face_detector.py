import cv2

from config.config import (
    FACE_SCALE_FACTOR,
    FACE_MIN_NEIGHBORS,
    FACE_MIN_SIZE
)


class FaceDetector:

    def __init__(
        self,
        scale_factor=FACE_SCALE_FACTOR,
        min_neighbors=FACE_MIN_NEIGHBORS,
        min_size=FACE_MIN_SIZE
    ):
        """
        Khởi tạo Face Detector.
        """

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.face_cascade = (
            cv2.CascadeClassifier(
                cascade_path
            )
        )

        if self.face_cascade.empty():

            raise RuntimeError(
                "Không thể tải Haar Cascade"
            )

        self.scale_factor = scale_factor

        self.min_neighbors = min_neighbors

        self.min_size = min_size

    def detect(self, frame):
        """
        Phát hiện khuôn mặt.

        Returns:
            danh sách bounding box.
        """

        if frame is None:
            return []

        # Chuyển sang grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Face detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        return faces

    def count_faces(self, faces):
        """
        Đếm số khuôn mặt.
        """

        return len(faces)

    def get_face_data(self, faces):
        """
        Chuẩn hóa dữ liệu khuôn mặt.
        """

        result = []

        for index, (x, y, w, h) in enumerate(faces):

            face = {
                "id": index + 1,
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            }

            result.append(face)

        return {
            "face_count": len(result),
            "faces": result
        }

    def draw_faces(
        self,
        frame,
        faces
    ):
        """
        Vẽ bounding box.
        """

        for index, (x, y, w, h) in enumerate(faces):

            # Bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Face ID
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