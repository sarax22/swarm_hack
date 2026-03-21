import cv2
import numpy as np

# -------- Find available cameras --------
def find_cameras(max_tested=10):
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
        cap.release()
    return available

camera_indices = find_cameras()

if len(camera_indices) == 0:
    print("No cameras found.")
    exit()

print("Available cameras:", camera_indices)

current_cam_idx = 0
cap = cv2.VideoCapture(camera_indices[current_cam_idx], cv2.CAP_DSHOW)

# -------- ArUco setup --------
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

print("\nControls:")
print("n → next camera")
print("p → previous camera")
print("q → quit\n")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    output = frame.copy()

    # ---- Detect markers ----
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(output, corners, ids)

        for corner, marker_id in zip(corners, ids):
            pts = corner.reshape((4, 2))
            cx = int((pts[0][0] + pts[2][0]) / 2)
            cy = int((pts[0][1] + pts[2][1]) / 2)

            cv2.circle(output, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(
                output,
                f"ID {marker_id[0]} ({cx},{cy})",
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    # ---- Show current camera index ----
    cv2.putText(
        output,
        f"Camera Index: {camera_indices[current_cam_idx]}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.imshow("Camera Selector + ArUco", output)

    key = cv2.waitKey(1) & 0xFF

    # ---- Controls ----
    if key == ord('q'):
        break

    elif key == ord('n'):  # next camera
        current_cam_idx = (current_cam_idx + 1) % len(camera_indices)
        cap.release()
        cap = cv2.VideoCapture(camera_indices[current_cam_idx], cv2.CAP_DSHOW)
        print(f"Switched to camera {camera_indices[current_cam_idx]}")

    elif key == ord('p'):  # previous camera
        current_cam_idx = (current_cam_idx - 1) % len(camera_indices)
        cap.release()
        cap = cv2.VideoCapture(camera_indices[current_cam_idx], cv2.CAP_DSHOW)
        print(f"Switched to camera {camera_indices[current_cam_idx]}")

cap.release()
cv2.destroyAllWindows()