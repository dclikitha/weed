
from ultralytics import YOLO
import cv2
import requests
import time

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = YOLO("best.pt")


# ==========================================
# API SETTINGS
# ==========================================

API_URL = "https://aislyntech.com/Api/55-update.php"

last_sent_value = None
last_send_time = 0

# Send at most once per second
SEND_INTERVAL = 1


# ==========================================
# SEND DETECTION STATUS TO API
# ==========================================

def send_detection_status(value):

    global last_sent_value
    global last_send_time

    current_time = time.time()

    # Don't repeatedly send the same value too quickly
    if (
        value == last_sent_value
        and (current_time - last_send_time) < SEND_INTERVAL
    ):
        return

    try:

        # IMPORTANT:
        # Use data= instead of json=
        response = requests.post(
            API_URL,
            data={
                "detected": str(value)
            },
            timeout=5
        )

        print(
            f"API SENT: {value} | "
            f"STATUS: {response.status_code} | "
            f"RESPONSE: {response.text}"
        )

        # Update only after request
        last_sent_value = value
        last_send_time = current_time

    except requests.exceptions.RequestException as e:

        print("API ERROR:", e)


# ==========================================
# OPEN SYSTEM CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Cannot open camera")
    exit()


print("Camera started...")
print("0 = WEED DETECTED")
print("1 = NO WEED DETECTED")
print("Press 'q' to quit.")


# ==========================================
# LIVE DETECTION
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Cannot read camera frame")
        break


    # ==========================================
    # YOLO PREDICTION
    # ==========================================

    results = model(
        frame,
        conf=0.25,
        imgsz=640,
        verbose=False
    )


    # Keep original camera frame
    output = frame.copy()

    # Assume no weed
    weed_detected = False


    # ==========================================
    # PROCESS YOLO DETECTIONS
    # ==========================================

    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Class ID
            class_id = int(box.cls[0])

            # Class name
            class_name = model.names[class_id]

            # Confidence
            confidence = float(box.conf[0])


            # ==========================================
            # ONLY PROCESS WEED
            # ==========================================

            if class_name.lower() == "weed":

                weed_detected = True

                # Bounding box
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )


                # Draw weed box
                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )


                # Weed label
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


    # ==========================================
    # DETECTION VALUE
    # ==========================================

    if weed_detected:

        # Weed detected
        detected_value = 0

    else:

        # No weed detected
        detected_value = 1


    # ==========================================
    # SEND TO API
    # ==========================================

    send_detection_status(detected_value)


    # ==========================================
    # DISPLAY STATUS
    # ==========================================

    if weed_detected:

        cv2.putText(
            output,
            "WEED DETECTED - 0",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            output,
            "NO WEED - 1",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # ==========================================
    # SHOW CAMERA
    # ==========================================

    cv2.imshow(
        "Weed Detection",
        output
    )


    # ==========================================
    # PRESS Q TO EXIT
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# RELEASE CAMERA
# ==========================================

cap.release()
cv2.destroyAllWindows()
