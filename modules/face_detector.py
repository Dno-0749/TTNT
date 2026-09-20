import os
import urllib.request
import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(
        self,
        min_detection_confidence=0.65
    ):
        """
        Khởi tạo Face Detector.
        Sử dụng MediaPipe Tasks API để tương thích với MediaPipe 0.10.x+.
        """
        self.min_detection_confidence = min_detection_confidence
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        self.model_path = os.path.join(models_dir, "blaze_face_short_range.tflite")
        if not os.path.exists(self.model_path):
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            try:
                print("Đang tải model FaceDetector...")
                urllib.request.urlretrieve(url, self.model_path)
            except Exception as e:
                print(f"Failed to download FaceDetector model: {e}")
                
        try:
            BaseOptions = mp.tasks.BaseOptions
            FaceDetectorTask = mp.tasks.vision.FaceDetector
            FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.IMAGE,
                min_detection_confidence=self.min_detection_confidence
            )
            self.detector = FaceDetectorTask.create_from_options(options)
        except Exception as e:
            print(f"Error initializing FaceDetector Task: {e}")
            self.detector = None

    def detect(self, frame):
        """
        Phát hiện tất cả khuôn mặt trong frame bằng MediaPipe Tasks API.

        Returns:
            Danh sách các bounding box:
            [
                (x, y, width, height),
                ...
            ]
        """
        if frame is None or self.detector is None:
            return []
        
        # Chuyển đổi BGR sang RGB cho MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        
        faces = []
        if detection_result and detection_result.detections:
            h, w, _ = frame.shape
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                
                # Bounding box của Tasks API trả về là pixel tuyệt đối
                x = int(bbox.origin_x)
                y = int(bbox.origin_y)
                width = int(bbox.width)
                height = int(bbox.height)
                
                # Cắt (clip) tọa độ để không bị văng ra khỏi viền camera
                x = max(0, x)
                y = max(0, y)
                width = min(w - x, width)
                height = min(h - y, height)
                
                faces.append((x, y, width, height))

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
                "NO FACE DETECTED",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        elif count > 1:
            cv2.putText(
                frame,
                f"WARNING: {count} FACES",
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
