import cv2


class FrameProcessor:

    def draw_info(
        self,
        frame,
        face_count,
        fps,
        detection_enabled,
        flip_enabled,
        recording
    ):
        """
        Vẽ thông tin hệ thống lên frame.
        """
        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        detection_status = (
            "ON"
            if detection_enabled
            else "OFF"
        )

        cv2.putText(
            frame,
            f"Detection: {detection_status}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        flip_status = (
            "ON"
            if flip_enabled
            else "OFF"
        )

        cv2.putText(
            frame,
            f"Flip: {flip_status}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        record_status = (
            "RECORDING"
            if recording
            else "OFF"
        )

        cv2.putText(
            frame,
            f"Recording: {record_status}",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255) if recording
            else (255, 255, 255),
            2
        )
        cv2.putText(
            frame,
            "[S] Photo  [R] Record  [D] Detect",
            (20, 440),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "[F] Flip  [Q] Quit",
            (20, 465),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        return frame