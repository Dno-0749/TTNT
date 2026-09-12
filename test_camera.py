import cv2

print("Dang mo camera...")

cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

if not cap.isOpened():
    print("KHONG MO DUOC CAMERA")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Khong doc duoc frame")
        break

    cv2.imshow("Test Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()