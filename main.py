import cv2
import time

from camera.camera_manager import CameraManager
from camera.video_recorder import VideoRecorder
from camera.screenshot import ScreenshotManager

from detection.face_detector import FaceDetector
from detection.face_tracker import FaceTracker

from processing.frame_processor import FrameProcessor

from utils.logger import setup_logger

from config.config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CAMERA_FPS,
    WINDOW_NAME,
    SCREENSHOT_FOLDER,
    RECORDING_FOLDER,
    KEY_QUIT,
    KEY_SCREENSHOT,
    KEY_RECORD,
    KEY_FLIP,
    KEY_DETECTION
)
def main():

    logger = setup_logger()

    camera = None
    recorder = None

    try:

        logger.info("Starting camera...")

        camera = CameraManager(
            camera_index=CAMERA_INDEX,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT
        )

        logger.info("Camera started successfully.")
        detector = FaceDetector()

        logger.info(
            "Face detector started successfully."
        )
        tracker = FaceTracker()

        logger.info(
            "Face tracker started successfully."
        )
        screenshot_manager = ScreenshotManager(
            SCREENSHOT_FOLDER
        )
        recorder = VideoRecorder(
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            fps=CAMERA_FPS,
            folder=RECORDING_FOLDER
        )
        processor = FrameProcessor()
        flip_enabled = True

        detection_enabled = True
        previous_time = time.time()

        fps = 0

        logger.info("Application started.")
        while True:

            frame = camera.get_frame(
                flip_enabled=flip_enabled
            )

            if frame is None:

                logger.error(
                    "Cannot read frame."
                )

                break
            current_time = time.time()

            elapsed_time = (
                current_time - previous_time
            )

            if elapsed_time > 0:

                fps = 1 / elapsed_time

            previous_time = current_time
            face_data = {
                "face_count": 0,
                "faces": []
            }

            if detection_enabled:

                faces = detector.detect(frame)
                frame = detector.draw_faces(
                    frame,
                    faces
                )

                face_data = detector.get_face_data(
                    faces
                )
                tracked_faces = tracker.update(
                    face_data["faces"]
                )

                face_data["faces"] = (
                    tracked_faces
                )

                face_data["face_count"] = (
                    len(tracked_faces)
                )
                for face in tracked_faces:

                    x = face["x"]
                    y = face["y"]

                    face_id = face["id"]

                    cv2.putText(
                        frame,
                        f"ID: {face_id}",
                        (x, y + face["height"] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )

            else:

                tracker.reset()
            frame = processor.draw_info(
                frame=frame,
                face_count=face_data["face_count"],
                fps=fps,
                detection_enabled=detection_enabled,
                flip_enabled=flip_enabled,
                recording=recorder.is_recording
            )
            if recorder.is_recording:

                recorder.write(frame)
            cv2.imshow(
                WINDOW_NAME,
                frame
            )
            key = cv2.waitKey(1) & 0xFF
            if key == KEY_QUIT:

                logger.info(
                    "Quit requested."
                )

                break
            elif key == KEY_SCREENSHOT:

                filepath = (
                    screenshot_manager.save(
                        frame
                    )
                )

                if filepath:

                    logger.info(
                        f"Screenshot saved: "
                        f"{filepath}"
                    )
            elif key == KEY_RECORD:

                try:

                    recording = (
                        recorder.toggle()
                    )

                    if recording:

                        logger.info(
                            "Recording started."
                        )

                    else:

                        logger.info(
                            "Recording stopped."
                        )

                except RuntimeError as error:

                    logger.error(
                        f"Recording error: {error}"
                    )
            elif key == KEY_FLIP:

                flip_enabled = not flip_enabled

                logger.info(
                    f"Flip: {flip_enabled}"
                )

            elif key == KEY_DETECTION:

                detection_enabled = (
                    not detection_enabled
                )

                logger.info(
                    f"Face detection: "
                    f"{detection_enabled}"
                )
    except RuntimeError as error:

        logger.error(
            f"Application error: {error}"
        )

    except KeyboardInterrupt:

        logger.info(
            "Application interrupted."
        )

    finally:

        if recorder is not None:

            recorder.release()

        if camera is not None:

            camera.release()

        cv2.destroyAllWindows()

        logger.info(
            "Application closed."
        )


if __name__ == "__main__":

    main()