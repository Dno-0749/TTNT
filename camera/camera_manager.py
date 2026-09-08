import cv2


class CameraManager:

    def __init__(
        self,
        camera_index=0,
        width=640,
        height=480
    ):
        """
        Khởi tạo Camera Manager.

        camera_index:
            0 = camera mặc định
            1 = camera thứ 2

        width:
            Chiều rộng frame.

        height:
            Chiều cao frame.
        """

        self.camera_index = camera_index
        self.width = width
        self.height = height

        # Mở camera
        self.cap = cv2.VideoCapture(
            self.camera_index
        )

        # Kiểm tra camera
        if not self.cap.isOpened():

            raise RuntimeError(
                f"Không thể mở camera "
                f"index={self.camera_index}"
            )

        # Thiết lập độ phân giải
        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height
        )

    def read(self):
        """
        Đọc một frame từ camera.

        Returns:
            frame nếu thành công.
            None nếu thất bại.
        """

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def resize(self, frame):
        """
        Resize frame.
        """

        return cv2.resize(
            frame,
            (
                self.width,
                self.height
            )
        )

    def flip(self, frame):
        """
        Lật camera theo chiều ngang.
        """

        return cv2.flip(frame, 1)

    def get_frame(self, flip_enabled=True):
        """
        Đọc và xử lý frame.

        Camera
            ↓
        Read
            ↓
        Resize
            ↓
        Flip
        """

        frame = self.read()

        if frame is None:
            return None

        frame = self.resize(frame)

        if flip_enabled:
            frame = self.flip(frame)

        return frame

    def release(self):
        """
        Giải phóng camera.
        """

        if self.cap is not None:

            self.cap.release()

    def is_opened(self):
        """
        Kiểm tra camera đang mở.
        """

        return self.cap is not None and self.cap.isOpened()