import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from prepare_data import prepare_data


X_train, X_test, y_train, y_test, scaler, encoder, features = prepare_data()

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "KNN": KNeighborsClassifier(n_neighbors=3),

    "SVM": SVC(),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = []

best_model = None
best_model_name = ""
best_f1 = 0

for name, model in models.items():
    print("\n-----------------------------")
    print("Model:", name)

    # Huấn luyện model
    model.fit(X_train, y_train)

    # Dự đoán dữ liệu test
    y_predict = model.predict(X_test)

    # Tính các chỉ số đánh giá
    accuracy = accuracy_score(y_test, y_predict)

    precision = precision_score(
        y_test,
        y_predict,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_predict,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_predict,
        average="weighted",
        zero_division=0
    )

    matrix = confusion_matrix(y_test, y_predict)

    print("Accuracy :", round(accuracy, 2))
    print("Precision:", round(precision, 2))
    print("Recall   :", round(recall, 2))
    print("F1-score :", round(f1, 2))
    print("Confusion Matrix:")
    print(matrix)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4)
    })

    # Chọn model có F1-score cao nhất
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name


# Lưu bảng so sánh kết quả
results_data = pd.DataFrame(results)
results_data.to_csv("model_results.csv", index=False)

print("\n=============================")
print("BẢNG SO SÁNH MODEL")
print(results_data)

print("\nModel tốt nhất:", best_model_name)


# Lưu model tốt nhất, scaler và encoder
model_data = {
    "model": best_model,
    "scaler": scaler,
    "encoder": encoder,
    "features": features
}

joblib.dump(model_data, "focus_model.pkl")

print("Đã lưu model: focus_model.pkl")
print("Đã lưu kết quả: model_results.csv")
