import joblib
import pandas as pd


def predict_state(
    ear,
    yaw,
    pitch,
    roll,
    blink_rate,
    eye_closure_duration,
    face_presence
):
    model_data = joblib.load("focus_model.pkl")

    model = model_data["model"]
    scaler = model_data["scaler"]
    encoder = model_data["encoder"]
    features = model_data["features"]

    input_data = pd.DataFrame(
        [[
            ear,
            yaw,
            pitch,
            roll,
            blink_rate,
            eye_closure_duration,
            face_presence
        ]],
        columns=features
    )

    input_data = scaler.transform(input_data)

    result_number = model.predict(input_data)

    result_label = encoder.inverse_transform(result_number)

    return result_label[0]


if __name__ == "__main__":
    state = predict_state(
        ear=0.30,
        yaw=2,
        pitch=1,
        roll=0,
        blink_rate=15,
        eye_closure_duration=0.10,
        face_presence=1
    )

    print("Trạng thái dự đoán:", state)
