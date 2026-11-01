import cv2
import requests

ESP32_IP = "10.100.45.158"

STREAM_URLS = [
    f"http://{ESP32_IP}:81/stream",
    f"http://{ESP32_IP}/stream",
    f"http://{ESP32_IP}/video",
    f"http://{ESP32_IP}/mjpeg"
]

cap = None
working_url = None

print("Checking ESP32-CAM...")

for url in STREAM_URLS:
    print("Testing:", url)

    try:
        response = requests.get(url, stream=True, timeout=3)

        content_type = response.headers.get("Content-Type", "")
        print("Status:", response.status_code)
        print("Content-Type:", content_type)

        if response.status_code == 200 and (
            "multipart" in content_type.lower()
            or "image" in content_type.lower()
        ):
            working_url = url
            response.close()
            break

        response.close()

    except Exception as e:
        print("Failed:", e)

if working_url is None:
    print()
    print("ERROR: Could not find ESP32-CAM video stream.")
    print("But the ESP32-CAM webpage is available at:")
    print(f"http://{ESP32_IP}/")
    exit()

print()
print("WORKING STREAM FOUND:")
print(working_url)
print()

cap = cv2.VideoCapture(working_url)

if not cap.isOpened():
    print("ERROR: OpenCV cannot open the stream.")
    exit()

print("ESP32-CAM stream connected!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot read camera frame")
        break

    cv2.imshow("ESP32-CAM Live Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Program stopped.")