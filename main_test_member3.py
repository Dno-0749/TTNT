"""
KỊCH BẢN KIỂM THỬ THỜI GIAN THỰC (REALTIME TEST SCRIPT) - THÀNH VIÊN 3
Dự án: Hệ thống AI đánh giá mức độ tập trung của sinh viên
Thành viên phụ trách: Thành viên 3 (Head Pose Estimation & Distraction Features)

Mục đích:
Chạy kiểm thử trực tiếp module head_pose.py từ Webcam hoặc sinh Frame thử nghiệm.
Hiển thị góc Yaw, Pitch, Roll và hướng nhìn FRONT, LEFT, RIGHT, UP, DOWN.
"""

import cv2
import sys
import numpy as np
from modules.head_pose import HeadPoseEstimator

# Cấu hình encoding stdout UTF-8 cho Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def run_webcam_test():
    """
    Chạy demo kiểm thử thời gian thực bằng Camera/Webcam.
    """
    print("=" * 60)
    print(" KHỞI ĐỘNG MODULE THÀNH VIÊN 3 - HEAD POSE ESTIMATION ")
    print("=" * 60)
    print("Hướng dẫn:")
    print(" - Nhìn thẳng vào màn hình camera -> NHÌN THẲNG (FRONT)")
    print(" - Quay mặt sang trái            -> QUAY TRÁI (LEFT)")
    print(" - Quay mặt sang phải            -> QUAY PHẢI (RIGHT)")
    print(" - Ngẩng đầu lên cao             -> NGẨNG ĐẦU (UP)")
    print(" - Cúi đầu xuống dưới            -> CÚI ĐẦU (DOWN)")
    print(" - Bấm phím 'q' trên cửa sổ hình ảnh để THOÁT.")
    print("=" * 60)

    estimator = HeadPoseEstimator(
        yaw_threshold=15.0,
        pitch_threshold=15.0
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[LỖI] Không thể kết nối với Webcam. Đang chuyển sang chế độ kiểm thử hình ảnh giả lập...")
        run_synthetic_test(estimator)
        return

    # Cấu hình độ phân giải camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[CẢNH BÁO] Không thể đọc frame từ camera!")
                break

            # Lật ảnh nằm ngang để camera hoạt động như gương (Mirror effect)
            frame = cv2.flip(frame, 1)

            # Xử lý frame bằng module HeadPoseEstimator của Thành viên 3
            annotated_frame, pose_info = estimator.process_frame(frame)

            # In thông tin console
            if pose_info["face_detected"]:
                print(
                    f"-> [THÀNH VIÊN 3 OUTPUT] Hướng nhìn: {pose_info['head_direction_vi']} ({pose_info['head_direction']}) | "
                    f"Yaw: {pose_info['yaw']}° | Pitch: {pose_info['pitch']}° | Roll: {pose_info['roll']}° | "
                    f"Xao nhãng tư thế: {pose_info['is_distracted_pose']}"
                )
            else:
                print(f"-> [THÀNH VIÊN 3 OUTPUT] {pose_info['head_direction_vi']}")

            # Hiển thị kết quả ra màn hình
            cv2.imshow("DEMO KIEM THU - THANH VIEN 3 (HEAD POSE)", annotated_frame)

            # Đợi phím nhấn 'q' để thoát
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[THÔNG BÁO] Đã nhận lệnh thoát chương trình.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Đã giải phóng camera và đóng tất cả cửa sổ.")


def run_synthetic_test(estimator: HeadPoseEstimator):
    """
    Chạy kiểm thử với ảnh màu tạo sẵn khi môi trường không có camera.
    """
    print("\n--- CHẠY KIỂM THỬ GIẢ LẬP VỚI KHUÔN MẶT MẪU ---")
    blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Thử nghiệm hàm process_frame
    annotated, result = estimator.process_frame(blank_image)
    print("Dữ liệu đầu ra của Thành viên 3 (Khi không có mặt):")
    print(result)

    # Thử nghiệm hàm process_from_landmarks với dữ liệu giả lập 468 điểm
    mock_landmarks = [(0.5, 0.5)] * 468
    # Thay đổi vài điểm mốc chính để mô phỏng nhìn thẳng (Mũi, Cằm, Mắt, Miệng)
    mock_landmarks[1] = (0.5, 0.5)      # Nose tip
    mock_landmarks[152] = (0.5, 0.8)    # Chin
    mock_landmarks[33] = (0.4, 0.4)     # Left eye
    mock_landmarks[263] = (0.6, 0.4)    # Right eye
    mock_landmarks[61] = (0.43, 0.65)   # Left mouth
    mock_landmarks[291] = (0.57, 0.65)  # Right mouth

    landmark_result = estimator.process_from_landmarks(mock_landmarks, (480, 640))
    print("\nDữ liệu đầu ra của Thành viên 3 (Khi nhận Landmark từ Người 1/2):")
    print(landmark_result)


if __name__ == "__main__":
    run_webcam_test()
