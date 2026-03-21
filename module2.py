import cv2
import numpy as np

# --- Config ---
DICT_TYPE = cv2.aruco.DICT_4X4_50
arucoDict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    bot_states = {}  # id -> {center, heading}

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, marker_id in enumerate(ids.flatten()):
            cx, cy  = get_marker_center(corners[i])
            heading = get_marker_heading(corners[i])
            bot_states[marker_id] = {"center": (cx, cy), "heading": heading}

            name = BOT_NAMES.get(marker_id, f"ID {marker_id}")
            label = f"{name} | {heading:.1f} deg"
            cv2.putText(frame, label, (cx + 10, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            

        font = cv2.FONT_HERSHEY_COMPLEX
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # Approximate and draw contour
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            cv2.drawContours(frame, [approx], 0, (0, 0, 255), 5)

            # Flatten points
            n = approx.ravel()
            i = 0
            for j in n:
                if i % 2 == 0:  # x, y coords
                    x, y = n[i], n[i + 1]
                    coord = f"{x} {y}"
                    if i == 0:  # first point
                        cv2.putText(frame, "Arrow tip", (x, y), font, 0.5, (255, 0, 0))
                    else:
                        cv2.putText(frame, coord, (x, y), font, 0.5, (0, 255, 0))
                i += 1

    return frame, bot_states

# --- Run on camera feed ---
cap = cv2.VideoCapture(0)  # change index if arena camera is on a different port

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, bot_states = process_frame(frame)

    # Print state for debugging / sending to bots
    for mid, state in bot_states.items():
        print(f"  ID {mid}: center={state['center']}, heading={state['heading']:.1f}°")

    cv2.imshow("Arena Camera - ArUco Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()