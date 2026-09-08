import cv2
class Camera:
    def __init__(self, camera_index=0, width=640, height=480):
        """
        Khởi tạo camera.

        camera_index:
            0 = webcam mặc định
            1 = camera thứ 2
            ...
        width, height:
            Kích thước frame sau khi resize.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Không thể mở camera với index = {self.camera_index}"
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        """
        Đọc một frame từ webcam.

        Returns:
            frame nếu đọc thành công.
            None nếu đọc thất bại (mất kết nối, camera bị rút...).
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def resize(self, frame):
        """
        Resize frame về kích thước mong muốn.
        """
        return cv2.resize(frame, (self.width, self.height))

    def flip(self, frame):
        """
        Lật frame theo chiều ngang.

        flipCode = 1:
            Lật ngang, tạo hiệu ứng giống gương.
        """
        return cv2.flip(frame, 1)

    def get_frame(self):
        """
        Đọc và xử lý một frame hoàn chỉnh.

        Quy trình:
        Webcam
           |
        Read frame
           |
        Resize
           |
        Flip
           |
        Return frame

        Returns:
            frame đã xử lý, hoặc None nếu đọc thất bại.
        """
        frame = self.read()
        if frame is None:
            return None

        frame = self.resize(frame)
        frame = self.flip(frame)
        return frame

    def is_opened(self):
        """
        Kiểm tra camera còn đang mở hay không.
        """
        return self.cap.isOpened()

    def release(self):
        """
        Giải phóng camera.
        """
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
if __name__ == "__main__":
    with Camera(camera_index=0) as cam:
        print("Nhấn 'q' để thoát...")
        while True:
            frame = cam.get_frame()
            if frame is None:
                print("Không đọc được frame, dừng chương trình.")
                break

            cv2.imshow("Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()