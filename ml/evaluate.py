import sys
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "dataset.csv")
    model_path = os.path.join(base_dir, "models", "focus_model.pkl")

    if not os.path.exists(model_path):
        print("❌ Chưa tìm thấy file focus_model.pkl. Hãy chạy ml/train_model.py trước!")
        return

    df = pd.read_csv(data_path)
    features = ["face_count", "ear", "blink_count", "closure_duration", "yaw", "pitch", "roll"]
    X = df[features]
    y_true = df["label"]

    model = joblib.load(model_path)
    y_pred = model.predict(X)

    # In báo cáo chi tiết
    labels = sorted(list(set(y_true)))
    print("\n" + "="*60)
    print(" BÁO CÁO ĐÁNH GIÁ CHI TIẾT (CLASSIFICATION REPORT)")
    print("="*60)
    print(classification_report(y_true, y_pred, labels=labels))

    # Vẽ Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - SmartStudy AI Model")
    plt.xlabel("Dự đoán (Predicted)")
    plt.ylabel("Thực tế (Actual)")
    plt.tight_layout()
    
    # Lưu biểu đồ ra file ảnh
    output_img = os.path.join(base_dir, "models", "confusion_matrix.png")
    plt.savefig(output_img)
    print(f"📊 Đã lưu biểu đồ đánh giá tại: {output_img}")
    plt.close()

if __name__ == "__main__":
    evaluate()