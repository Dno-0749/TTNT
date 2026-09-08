import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, Optional, List


class HeadPoseEstimator:
    """
    Lớp xử lý Ước lượng Tư thế Đầu (Head Pose Estimation) đa nền tảng,
    tự động chọn backend MediaPipe FaceLandmarker Task, Legacy FaceMesh hoặc OpenCV Cascade Fallback.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        yaw_threshold: float = 15.0,
        pitch_threshold: float = 15.0
    ):
        """
        Khởi tạo mô hình HeadPoseEstimator.
        """
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold

        # Đường dẫn thư mục mô hình
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.model_path = os.path.join(project_root, "models", "face_landmarker.task")
        self.cascade_path = os.path.join(project_root, "models", "haarcascade_frontalface_default.xml")

        # 1. Khởi tạo Backend MediaPipe FaceLandmarker Tasks API (Chuẩn mới MediaPipe 1.0+)
        self.landmarker = None
        self.legacy_face_mesh = None

        if os.path.exists(self.model_path):
            try:
                BaseOptions = mp.tasks.BaseOptions
                FaceLandmarker = mp.tasks.vision.FaceLandmarker
                FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode

                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=self.model_path),
                    running_mode=VisionRunningMode.IMAGE,
                    num_faces=max_num_faces,
                    min_face_detection_confidence=min_detection_confidence
                )
                self.landmarker = FaceLandmarker.create_from_options(options)
                print("[HEAD POSE INFO] Da khoi tao thanh cong MediaPipe FaceLandmarker Task API.")
            except Exception as e:
                print(f"[HEAD POSE CANH BAO] Khong the khoi tao Task API: {e}")
                self.landmarker = None

        # 2. Khởi tạo Backend MediaPipe Legacy Solutions (Nếu phiên bản có hỗ trợ mp.solutions)
        if self.landmarker is None and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
            try:
                self.legacy_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=static_image_mode,
                    max_num_faces=max_num_faces,
                    refine_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                print("[HEAD POSE INFO] Đã khởi tạo MediaPipe Legacy FaceMesh API.")
            except Exception:
                self.legacy_face_mesh = None

        # 3. Khởi tạo Backup HaarCascade Classifier
        if os.path.exists(self.cascade_path):
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        else:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Mô hình 3D khuôn mặt chuẩn trong không gian thế giới (3D World Coordinates in mm)
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # Mũi (Nose tip) - Index 1
            (0.0, -330.0, -65.0),        # Cằm (Chin) - Index 152
            (-225.0, 170.0, -135.0),     # Khóe mắt trái (Left eye corner) - Index 33
            (225.0, 170.0, -135.0),      # Khóe mắt phải (Right eye corner) - Index 263
            (-150.0, -150.0, -125.0),    # Khóe miệng trái (Left mouth corner) - Index 61
            (150.0, -150.0, -125.0)      # Khóe miệng phải (Right mouth corner) - Index 291
        ], dtype=np.float64)

        # Các chỉ số Landmark tương ứng trong MediaPipe 468 Face Mesh
        self.landmark_indices = [1, 152, 33, 263, 61, 291]

        # Từ điển dịch hướng nhìn sang Tiếng Việt hiển thị
        self.direction_vi_map = {
            "FRONT": "NHÌN THẲNG",
            "LEFT": "QUAY TRÁI",
            "RIGHT": "QUAY PHẢI",
            "UP": "NGẨNG ĐẦU",
            "DOWN": "CÚI ĐẦU",
            "UNKNOWN": "KHÔNG XÁC ĐỊNH"
        }

    def estimate_pose_from_2d_points(
        self,
        image_points_2d: np.ndarray,
        img_size: Tuple[int, int]
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Tính toán góc xoay Euler (Yaw, Pitch, Roll) từ 6 tọa độ 2D của các điểm mốc khuôn mặt.
        """
        height, width = img_size

        focal_length = width
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(
            self.model_points_3d,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0, np.zeros((3, 1)), np.zeros((3, 1))

        rot_mat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]

        # Chuẩn hóa góc về dải [-90, 90] độ để tâm 0.0° trùng với hướng nhìn thẳng
        if pitch > 90.0:
            pitch = pitch - 180.0
        elif pitch < -90.0:
            pitch = pitch + 180.0

        # Đảo chiều dấu Pitch để Cúi đầu = Pitch âm (-), Ngẩng đầu = Pitch dương (+)
        pitch = -pitch

        if yaw > 90.0:
            yaw = yaw - 180.0
        elif yaw < -90.0:
            yaw = yaw + 180.0

        # Đảo chiều dấu Yaw để Quay trái = Yaw âm (-), Quay phải = Yaw dương (+) theo góc nhìn gương camera
        yaw = -yaw

        return yaw, pitch, roll, rot_vec, trans_vec

    def classify_head_direction(self, yaw: float, pitch: float) -> str:
        """
        Phân loại hướng nhìn của sinh viên dựa trên góc Yaw và Pitch.
        """
        if yaw > self.yaw_threshold:
            return "RIGHT"
        elif yaw < -self.yaw_threshold:
            return "LEFT"
        
        if pitch > self.pitch_threshold:
            return "UP"
        elif pitch < -self.pitch_threshold:
            return "DOWN"

        return "FRONT"

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Hàm chính nhận đầu vào là Frame ảnh camera, xử lý ước lượng tư thế và vẽ kết quả lên frame.
        """
        height, width, _ = frame.shape
        annotated_frame = frame.copy()
        image_points_2d = None

        # 1. Thử phát hiện bằng MediaPipe FaceLandmarker Task API
        if self.landmarker is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.landmarker.detect(mp_image)

            if detection_result and detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                pts = []
                for idx in self.landmark_indices:
                    lm = landmarks[idx]
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    pts.append([cx, cy])
                image_points_2d = np.array(pts, dtype=np.float64)

        # 2. Thử phát hiện bằng Legacy MediaPipe FaceMesh nếu Task API chưa có kết quả
        if image_points_2d is None and self.legacy_face_mesh is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.legacy_face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                pts = []
                for idx in self.landmark_indices:
                    lm = face_landmarks.landmark[idx]
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    pts.append([cx, cy])
                image_points_2d = np.array(pts, dtype=np.float64)

        # 3. Fallback OpenCV HaarCascade nếu cả 2 phương pháp MediaPipe không thấy khuôn mặt
        if image_points_2d is None and hasattr(self, 'face_cascade') and not self.face_cascade.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                pts = [
                    [x + w * 0.5, y + h * 0.55],   # Mũi
                    [x + w * 0.5, y + h * 0.95],   # Cằm
                    [x + w * 0.25, y + h * 0.35],  # Mắt trái
                    [x + w * 0.75, y + h * 0.35],  # Mắt phải
                    [x + w * 0.3, y + h * 0.75],   # Miệng trái
                    [x + w * 0.7, y + h * 0.75]    # Miệng phải
                ]
                image_points_2d = np.array(pts, dtype=np.float64)

        default_output = {
            "face_detected": False,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "head_direction": "UNKNOWN",
            "head_direction_vi": "KHÔNG THẤY KHUÔN MẶT",
            "is_distracted_pose": True,
            "message": "Không phát hiện khuôn mặt sinh viên"
        }

        if image_points_2d is None:
            cv2.putText(
                annotated_frame,
                "TRANG THAI: KHONG THAY KHUON MAT",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            return annotated_frame, default_output

        # Tính toán tư thế đầu (Head Pose)
        yaw, pitch, roll, rot_vec, trans_vec = self.estimate_pose_from_2d_points(
            image_points_2d, (height, width)
        )

        direction = self.classify_head_direction(yaw, pitch)
        direction_vi = self.direction_vi_map.get(direction, "KHÔNG XÁC ĐỊNH")
        is_distracted = (direction != "FRONT")

        # Vẽ đường vector định hướng từ mũi
        nose_2d = tuple(image_points_2d[0].astype(int))
        focal_length = width
        camera_matrix = np.array([
            [focal_length, 0, width / 2.0],
            [0, focal_length, height / 2.0],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        nose_end_point_3d = np.array([(0.0, 0.0, 500.0)], dtype=np.float64)
        nose_end_point_2d, _ = cv2.projectPoints(
            nose_end_point_3d, rot_vec, trans_vec, camera_matrix, dist_coeffs
        )
        p1 = nose_2d
        p2 = (int(nose_end_point_2d[0][0][0]), int(nose_end_point_2d[0][0][1]))

        line_color = (0, 255, 0) if direction == "FRONT" else (0, 0, 255)
        cv2.line(annotated_frame, p1, p2, line_color, 3)

        for pt in image_points_2d:
            cv2.circle(annotated_frame, (int(pt[0]), int(pt[1])), 3, (255, 255, 0), -1)

        # Hiển thị bảng điều khiển thông số
        cv2.rectangle(annotated_frame, (20, 20), (460, 160), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (20, 20), (460, 160), line_color, 2)

        cv2.putText(
            annotated_frame,
            f"HUONG NHIN: {direction_vi} ({direction})",
            (35, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            line_color,
            2
        )
        cv2.putText(
            annotated_frame,
            f"Goc Yaw (Xoay Trai/Phai): {yaw:.1f} deg",
            (35, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )
        cv2.putText(
            annotated_frame,
            f"Goc Pitch (Gat/Ngang):   {pitch:.1f} deg",
            (35, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )
        cv2.putText(
            annotated_frame,
            f"Goc Roll (Nghieng dau):   {roll:.1f} deg",
            (35, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        feature_dict = {
            "face_detected": True,
            "yaw": round(float(yaw), 2),
            "pitch": round(float(pitch), 2),
            "roll": round(float(roll), 2),
            "head_direction": direction,
            "head_direction_vi": direction_vi,
            "is_distracted_pose": is_distracted,
            "message": f"Sinh viên đang {direction_vi.lower()}"
        }

        return annotated_frame, feature_dict

    def process_from_landmarks(
        self,
        landmarks_list: List[Tuple[float, float]],
        img_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Hàm giao tiếp mở rộng: Nhận danh sách tọa độ landmark trực tiếp từ Module Người 1 / Người 2.
        """
        if not landmarks_list or len(landmarks_list) < max(self.landmark_indices):
            return {
                "face_detected": False,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "head_direction": "UNKNOWN",
                "head_direction_vi": "KHÔNG ĐỦ ĐIỂM LANDMARK",
                "is_distracted_pose": True
            }

        height, width = img_size
        image_points_2d = []
        for idx in self.landmark_indices:
            pt = landmarks_list[idx]
            px = int(pt[0] * width) if pt[0] <= 1.0 else int(pt[0])
            py = int(pt[1] * height) if pt[1] <= 1.0 else int(pt[1])
            image_points_2d.append([px, py])

        image_points_2d = np.array(image_points_2d, dtype=np.float64)
        yaw, pitch, roll, _, _ = self.estimate_pose_from_2d_points(image_points_2d, (height, width))
        direction = self.classify_head_direction(yaw, pitch)

        return {
            "face_detected": True,
            "yaw": round(float(yaw), 2),
            "pitch": round(float(pitch), 2),
            "roll": round(float(roll), 2),
            "head_direction": direction,
            "head_direction_vi": self.direction_vi_map.get(direction, "KHÔNG XÁC ĐỊNH"),
            "is_distracted_pose": (direction != "FRONT")
        }