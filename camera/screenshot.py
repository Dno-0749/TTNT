import cv2
import os
from datetime import datetime


class ScreenshotManager:

    def __init__(self, folder="captures"):

        self.folder = folder

        # Tạo folder nếu chưa tồn tại
        os.makedirs(
            self.folder,
            exist_ok=True
        )

    def save(self, frame):
        """
        Lưu frame hiện tại thành ảnh.

        Returns:
            Đường dẫn file ảnh.
        """

        if frame is None:
            return None

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = (
            f"photo_{timestamp}.jpg"
        )

        filepath = os.path.join(
            self.folder,
            filename
        )

        success = cv2.imwrite(
            filepath,
            frame
        )

        if success:
            return filepath

        return None