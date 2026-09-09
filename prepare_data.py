import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def prepare_data():
    data = pd.read_csv("dataset.csv")

    features = [
        "ear",
        "yaw",
        "pitch",
        "roll",
        "blink_rate",
        "eye_closure_duration",
        "face_presence"
    ]

    # Nếu có ô trống thì thay bằng 0
    data = data.fillna(0)

    X = data[features]
    y = data["label"]

    # Đổi label chữ thành số
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Chia 80% train và 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, encoder, features


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, encoder, features = prepare_data()

    print("Đã xử lý dữ liệu thành công.")
    print("Số mẫu train:", len(X_train))
    print("Số mẫu test:", len(X_test))
    print("Các trạng thái:", encoder.classes_)
