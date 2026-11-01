from ultralytics import YOLO
import cv2
import requests
import time

# Load trained model
model = YOLO(r"C:\Users\likhi\OneDrive\Desktop\weed\weed\best.pt")


import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# API settings from .env
API_URL = os.getenv("API_URL")
SEND_INTERVAL = float(os.getenv("SEND_INTERVAL", "1"))

# ESP32-CAM IP from .env
ESP32_IP = os.getenv("ESP32_IP")

if not API_URL:
    raise ValueError("API_URL is missing from .env")

if not ESP32_IP:
    raise ValueError("ESP32_IP is missing from .env")

last_sent_value = None
last_send_time = 0

# Stream URLs
STREAM_URLS = [
    f"http://{ESP32_IP}:81/stream",
    f"http://{ESP32_IP}/stream"
]

def send_detection_status(value):
    global last_sent_value, last_send_time

    current_time = time.time()

    if value == last_sent_value and (current_time - last_send_time) < SEND_INTERVAL:
        return

    try:
        response = requests.post(
            API_URL,
            data={"detected": str(value)},
            timeout=5
        )

        print(
            f"API SENT: {value} | "
            f"STATUS: {response.status_code} | "
            f"RESPONSE: {response.text}"
        )

        last_sent_value = value
        last_send_time = current_time

    except requests.exceptions.RequestException as e:
        print("API ERROR:", e)

# Find working ESP32-CAM stream
print("Connecting to ESP32-CAM...")
print("ESP32 IP:", ESP32_IP)

cap = None
working_url = None

for url in STREAM_URLS:
    print("Trying:", url)

    test_cap = cv2.VideoCapture(url)

    if test_cap.isOpened():
        ret, frame = test_cap.read()

        if ret and frame is not None:
            cap = test_cap
            working_url = url
            print("Working stream found:", url)
            break

    test_cap.release()

if cap is None:
    print("ERROR: Cannot connect to ESP32-CAM")
    print("Camera webpage is working at:")
    print(f"http://{ESP32_IP}/")
    print("But OpenCV cannot find the video stream.")
    exit()

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("ESP32-CAM connected!")
print("Stream URL:", working_url)
print("0 = WEED DETECTED")
print("1 = NO WEED DETECTED")
print("Press 'q' to quit.")

# Create camera window
WINDOW_NAME = "ESP32-CAM Weed Detection"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1200, 800)

# Live detection
while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot read ESP32-CAM frame")
        cap.release()
        time.sleep(1)

        cap = cv2.VideoCapture(working_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        continue

    # YOLO prediction
    results = model(
        frame,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    output = frame.copy()
    weed_detected = False

    # Process detections
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if class_name.lower() == "weed":
                weed_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                label = f"WEED {confidence * 100:.1f}%"

                cv2.putText(
                    output,
                    label,
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

    # Detection value
    detected_value = 1 if weed_detected else 0

    # Send value to API
    send_detection_status(detected_value)

    # Display status
    if weed_detected:
        cv2.putText(
            output,
            "WEED DETECTED - 1",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
    else:
        cv2.putText(
            output,
            "NO WEED - 0",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Show camera
    cv2.imshow(WINDOW_NAME, output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Program stopped.")