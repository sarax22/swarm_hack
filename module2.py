import cv2 as cv
import numpy as np

# --- Config ---
DICT_TYPE = cv.aruco.DICT_4X4_50
arucoDict = cv.aruco.getPredefinedDictionary(DICT_TYPE)
arucoParams = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)

# Map marker IDs to bot names (agree this with other teams!)
BOT_NAMES = {0: "Bot_A", 1: "Bot_B", 2: "Bot_C"}

def get_marker_center(corners_single):
    """Returns (x, y) pixel center of a single marker."""
    pts = corners_single[0]  # shape (4, 2)
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    return cx, cy

def get_marker_heading(corners_single):
    """Estimate heading from top-edge midpoint vs centre."""
    pts = corners_single[0]
    top_mid = ((pts[0] + pts[1]) / 2)
    center  = np.mean(pts, axis=0)
    delta   = top_mid - center
    angle   = np.degrees(np.arctan2(-delta[1], delta[0]))  # screen coords
    return angle

def process_frame(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    bot_states = {}  # id -> {center, heading}

    if ids is not None:
        cv.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, marker_id in enumerate(ids.flatten()):
            cx, cy  = get_marker_center(corners[i])
            heading = get_marker_heading(corners[i])
            bot_states[marker_id] = {"center": (cx, cy), "heading": heading}

            name = BOT_NAMES.get(marker_id, f"ID {marker_id}")
            label = f"{name} | {heading:.1f} deg"
            cv.putText(frame, label, (cx + 10, cy),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, bot_states

# --- Run on camera feed ---
cap = cv.VideoCapture(0)  # change index if arena camera is on a different port

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, bot_states = process_frame(frame)

    # Print state for debugging / sending to bots
    for mid, state in bot_states.items():
        print(f"  ID {mid}: center={state['center']}, heading={state['heading']:.1f}°")

    cv.imshow("Arena Camera - ArUco Tracking", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()