import cv2
import os
from datetime import datetime


class VideoRecorder:

    def __init__(
        self,
        width=640,
        height=480,
        fps=30,
        folder="recordings"
    ):
        """
        Khởi tạo Video Recorder.
        """

        self.width = width
        self.height = height
        self.fps = fps
        self.folder = folder

        self.writer = None

        self.is_recording = False

        self.filepath = None

        # Tạo folder
        os.makedirs(
            self.folder,
            exist_ok=True
        )

    def start(self):
        """
        Bắt đầu quay video.
        """

        if self.is_recording:
            return

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = (
            f"video_{timestamp}.mp4"
        )

        self.filepath = os.path.join(
            self.folder,
            filename
        )

        # Codec MP4
        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        self.writer = cv2.VideoWriter(
            self.filepath,
            fourcc,
            self.fps,
            (
                self.width,
                self.height
            )
        )

        if not self.writer.isOpened():

            self.writer = None

            raise RuntimeError(
                "Không thể khởi tạo VideoWriter"
            )

        self.is_recording = True

    def write(self, frame):
        """
        Ghi một frame vào video.
        """

        if not self.is_recording:
            return

        if self.writer is None:
            return

        if frame is None:
            return

        self.writer.write(frame)

    def stop(self):
        """
        Dừng quay video.
        """

        if not self.is_recording:
            return self.filepath

        if self.writer is not None:

            self.writer.release()

        self.writer = None

        self.is_recording = False

        return self.filepath

    def toggle(self):
        """
        Bật / tắt recording.

        Returns:
            True  = đang quay
            False = đã dừng
        """

        if self.is_recording:

            self.stop()

            return False

        self.start()

        return True

    def release(self):
        """
        Giải phóng VideoWriter.
        """

        if self.is_recording:

            self.stop()