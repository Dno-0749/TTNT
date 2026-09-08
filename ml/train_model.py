import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def train_and_compare():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print("❌ Chưa có file dataset.csv!")
        return

    df = pd.read_csv(data_path)
    X = df[["face_count", "ear", "blink_count", "closure_duration", "yaw", "pitch", "roll"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='rbf', probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = []
    best_f1 = 0
    best_model = None
    best_name = ""

    print("\n" + "="*60)
    print(" BẢNG SO SÁNH CÁC MÔ HÌNH MACHINE LEARNING (BÁO CÁO NGƯỜI 4)")
    print("="*60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1})
        print(f"| {name:<15} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} |")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    # Lưu model tốt nhất
    model_save_path = os.path.join(models_dir, "focus_model.pkl")
    joblib.dump(best_model, model_save_path)
    print("="*60)
    print(f"🏆 Mô hình tốt nhất: {best_name} (F1: {best_f1:.4f}) -> Đã lưu vào {model_save_path}\n")

if __name__ == "__main__":
    train_and_compare()