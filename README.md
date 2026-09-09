# 🎓 SMARTSTUDY AI - AI Student Focus Monitoring System

**SMARTSTUDY AI** là hệ thống giám sát và đánh giá mức độ tập trung của sinh viên theo thời gian thực (Real-time Focus Monitoring System) dựa trên các kỹ thuật Xử lý ảnh (Computer Vision) và Học máy (Machine Learning).

---

## 🌟 Tính Năng Chính (Key Features)

- **📷 Real-time Camera Stream:** Luồng theo dõi camera trực tiếp với tốc độ phản hồi cao.
- **👁️ Eye Aspect Ratio (EAR) & Sleepiness Detection:** Theo dõi nhắm/mở mắt, đếm số lần chớp mắt và phát hiện trạng thái ngủ gật.
- **🗣️ Head Pose Estimation (Yaw/Pitch/Roll):** Ước lượng hướng tư thế đầu (Nhìn thẳng, Xoay trái/phải, Ngẩng/Cúi).
- **🤖 Machine Learning Classification:** Phân loại trạng thái học tập (`Focused`, `Distracted`, `Sleepy`, `Absent`) sử dụng mô hình Random Forest.
- **⏱️ Temporal Analysis & State Machine:** Lọc nhiễu chuyển động ngắn và quản lý trạng thái chuyển đổi thông minh (delay 1.5s - warning 4s).
- **📊 Real-time Dashboard & Session Analytics:** Giao diện Streamlit hiển thị chỉ số tập trung (Focus Score 0-100), cảnh báo tự động và báo cáo tổng kết phiên học.

---

## 🏗️ Cấu Trúc Dự Án (Project Structure)

```text
SMARTSTUDY_AI/
├── app.py                      # File khởi chạy chính (Streamlit Web App)
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── README.md                   # Tài liệu hướng dẫn dự án
├── dashboard/                  # Các module giao diện Streamlit
│   ├── analytics.py            # Báo cáo phân tích chuyên sâu & khuyến nghị
│   └── realtime.py             # Dashboard hiển thị chỉ số thời gian thực
├── data/                       # Quản lý bộ dữ liệu
│   ├── data_manager.py         # Cập nhật & đọc trạng thái dữ liệu phiên học
│   └── dataset.csv             # Bộ dữ liệu huấn luyện đã thu thập
├── ml/                         # Luồng huấn luyện & Đánh giá Machine Learning
│   ├── prepare_data.py         # Script thu thập & tạo Dataset từ camera
│   ├── train_model.py          # Script huấn luyện & so sánh 4 mô hình ML
│   └── evaluate.py             # Script đánh giá & xuất Ma trận nhầm lẫn
├── models/                     # Thư mục chứa các file Mô hình & Cascade
│   ├── confusion_matrix.png    # Biểu đồ đánh giá ma trận nhầm lẫn
│   ├── face_landmarker.task    # Model MediaPipe Face Landmarker
│   ├── focus_model.pkl         # Mô hình Machine Learning đã trained
│   └── haarcascade_*.xml       # File Haar Cascade nhận diện khuôn mặt
└── modules/                    # Xử lý lõi hệ thống (Core Processing)
    ├── camera.py               # Quản lý luồng Camera & OpenCV
    ├── eye_analyzer.py         # Phân tích thông số mắt & tính EAR
    ├── face_detector.py        # Nhận diện khuôn mặt & tạo Bounding Box
    ├── focus_engine.py         # Thuật toán tính Focus Score & Temporal Analysis
    ├── head_pose.py            # Ước lượng tư thế đầu (Yaw, Pitch, Roll)
    └── session_tracker.py      # Theo dõi thời lượng & thống kê phiên học
🛠️ Cài Đặt & Hướng Dẫn Sử Dụng (Installation & Usage)
1. Yêu cầu hệ thống
Python 3.9 trở lên

Webcam tích hợp hoặc Camera gắn ngoài

2. Cài đặt môi trường
Bash
# Clone repository
git clone [https://github.com/Dno-0749/TTNT.git](https://github.com/Dno-0749/TTNT.git)
cd TTNT

# Khởi tạo môi trường ảo Python
python -m venv .venv

# Kích hoạt môi trường ảo (Windows Command Prompt)
.venv\Scripts\activate

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
3. Huấn luyện mô hình ML (Tùy chọn)
Bash
# Thu thập thêm dữ liệu (nếu muốn)
python ml/prepare_data.py

# Huấn luyện và so sánh các mô hình ML
python ml/train_model.py

# Xuất biểu đồ ma trận nhầm lẫn
python ml/evaluate.py
4. Khởi chạy ứng dụng Web Dashboard
Bash
streamlit run app.py
Trình duyệt sẽ tự động mở địa chỉ http://localhost:8501. Tích chọn "Start Camera" để bắt đầu trải nghiệm!

📊 Kết Quả Huấn Luyện Mô Hình (ML Evaluation)
Dự án thử nghiệm và so sánh 4 thuật toán Machine Learning (Decision Tree, KNN, SVM, Random Forest). Mô hình Random Forest đạt hiệu năng và độ ổn định cao nhất trên tập dữ liệu kiểm thử.
# TTNT
