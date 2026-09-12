# TTNT

## Cấu trúc dự án

```text
TTNT/
|-- main.py
|-- modules/
|   |-- camera.py
|   |-- face_detector.py
|   |-- head_pose.py
|   |-- eye_analyzer.py
|   `-- __init__.py
|-- tests/
`-- README.md
```

## Eye Tracking & Sleepiness Detection

Module `modules/eye_analyzer.py` phân tích 468 facial landmarks của MediaPipe:

- Xác định mắt trái/phải theo landmark chuẩn MediaPipe.
- Tính Eye Aspect Ratio (EAR) cho từng mắt và EAR trung bình.
- Phân loại `OPEN`/`CLOSED`, đếm chớp mắt và đo thời gian nhắm mắt.
- Đánh dấu `is_sleepy=True` khi cả hai mắt đóng lâu hơn `sleepy_duration`.

Ví dụ nối với Face Mesh hiện có:

```python
from modules.eye_analyzer import EyeAnalyzer

eye_analyzer = EyeAnalyzer(
	ear_threshold=0.21,
	sleepy_duration=2.0,
	max_blink_duration=0.8,
)
eye_result = eye_analyzer.process_landmarks(face_landmarks, timestamp=timestamp)
```

`face_landmarks` có thể là danh sách `(x, y)`, dictionary `{"x": ..., "y": ...}`
hoặc object MediaPipe có thuộc tính `.x`, `.y`. `timestamp` tính bằng giây và nên
là giá trị monotonic theo từng frame.

Chạy test:

```powershell
python -m unittest discover -s tests -v
```

Các trường hợp cần kiểm thử camera thực tế: mắt mở, mắt nhắm, chớp mắt, nhắm mắt
lâu, ánh sáng yếu/thay đổi và người đeo kính. Với ánh sáng hoặc kính làm EAR dao
động, cần hiệu chỉnh `ear_threshold` theo dữ liệu thực tế thay vì dùng cứng `0.21`.