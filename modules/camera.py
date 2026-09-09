import sys
import threading
import time
import cv2


class Camera:
    def __init__(self, camera_index=0, width=640, height=480):
        """
        Khởi tạo Camera đa luồng (Threaded Zero-Latency Capture).
        Khắc phục hoàn toàn hiện tượng trễ tích lũy và đơ cam sau một thời gian chạy.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height

        # 1. Khởi tạo VideoCapture: Ưu tiên backend DirectShow trên Windows
        self.cap = None
        if sys.platform == "win32":
            try:
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            except Exception:
                self.cap = None

        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Không thể mở camera với index = {self.camera_index}"
            )

        # 2. Cấu hình phần cứng: Buffer size = 1 để chống trễ tích lũy
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # 3. Khởi tạo Thread đọc camera liên tục trong background
        self.latest_frame = None
        self.is_running = True
        self.lock = threading.Lock()

        # Đọc 1 frame mồi trước khi kích hoạt thread
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.latest_frame = frame

        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()

    def _capture_worker(self):
        """
        Thread nền liên tục đọc frame từ phần cứng webcam,
        đảm bảo OS buffer luôn được xả sạch, chống đơ và lag tuyệt đối.
        """
        consecutive_failures = 0
        while self.is_running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                consecutive_failures = 0
                with self.lock:
                    self.latest_frame = frame
            else:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    # Mất kết nối camera quá lâu
                    time.sleep(0.05)
                else:
                    time.sleep(0.01)

            # Nhường CPU ngắn để tránh chiếm 100% tài nguyên một nhân CPU
            time.sleep(0.003)

    def read(self):
        """
        Lấy frame mới nhất hiện có với độ trễ xấp xỉ 0ms.
        """
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def resize(self, frame):
        return cv2.resize(frame, (self.width, self.height))

    def flip(self, frame):
        return cv2.flip(frame, 1)

    def get_frame(self):
        frame = self.read()
        if frame is None:
            return None

        frame = self.resize(frame)
        frame = self.flip(frame)
        return frame

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened() and self.is_running

    def release(self):
        self.is_running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=0.3)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

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