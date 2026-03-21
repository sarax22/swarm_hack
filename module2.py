import cv2
import numpy as np
from astar import astar
import mss
import time
import matplotlib.pyplot as plt
from capture import setup_camera

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
    grid = 0
    path1 = path2 = path3 = []
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
            
        marker_centers = []
        for i in list(bot_states.values()):
            marker_centers.append(i["center"])
        print(bot_states,marker_centers)
                    

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 150, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 80])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2

        # ---- EDGE detection instead of grayscale threshold ----
        edges = cv2.Canny(red_mask, 50, 150)

        cv2.imshow("EDGE MASK", edges)
        cv2.waitKey(0)

        # ---- Contours from edges ----
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        HEIGHT, WIDTH = edges.shape  # use your actual image size
        CHUNK = 5

        CHUNK_H = HEIGHT // CHUNK
        CHUNK_W = WIDTH // CHUNK

        grid = np.zeros((CHUNK_H, CHUNK_W), dtype=int)

        max_y, max_x, _ = frame.shape
        mask = np.zeros_like(edges)


        cv2.imshow("RED MASK", red_mask)
        cv2.waitKey(0)


        # Draw only red contours
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

            # draw on the output-visible image
            cv2.drawContours(frame, [approx], -1, (0, 0, 255), 5)

            # draw ONLY this contour on mask
            cv2.drawContours(mask, [cnt], -1, 255, 1)

        # Exclude ArUco areas (run ONCE)
        EXCLUDE_RADIUS = 60
        for (mx, my) in marker_centers:
            cv2.circle(mask, (mx, my), EXCLUDE_RADIUS, 0, -1)

        # Build occupancy grid (run ONCE)
        ys, xs = np.where(mask == 255)

        for x, y in zip(xs, ys):
            cx = min(x // CHUNK, CHUNK_W - 1)
            cy = min(y // CHUNK, CHUNK_H - 1)
            grid[cy, cx] = 1

        inflate_radius = 3 
        grid_img = (grid.astype(np.uint8)) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflate_radius*2+1, inflate_radius*2+1))
        inflated = cv2.dilate(grid_img, kernel)
        inflated_grid = (inflated > 0).astype(int)

        start1 = (bot_states[4]["center"][1]//CHUNK,bot_states[4]["center"][0]//CHUNK)
        start2 = (bot_states[2]["center"][1]//CHUNK,bot_states[2]["center"][0]//CHUNK)
        start3 = (bot_states[3]["center"][1]//CHUNK,bot_states[3]["center"][0]//CHUNK)
        goal = (bot_states[7]["center"][1]//CHUNK,bot_states[7]["center"][0]//CHUNK)


        path1 = astar(inflated_grid, start1, goal)
        path2 = astar(inflated_grid, start2, goal)
        path3 = astar(inflated_grid, start3, goal)


    return frame, bot_states, grid, path1, path2, path3

# --- Run on camera feed ---
# with mss.mss() as sct:
#     monitor = {
#         "top": 0,     # adjust to your app
#         "left": 0,
#         "width": 1920,
#         "height": 1200
#     }
#     time.sleep(1)
#     while True:
#         # Replace cap.read()
#         screenshot = sct.grab(monitor)
#         frame = np.array(screenshot)

#         # Convert BGRA → BGR (IMPORTANT)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

#         # Your existing pipeline
#         frame, bot_states, grid, path1, path2, path3 = process_frame(frame)
#         print(path1)
#         print()
#         print(path2)
#         print()
#         print(path3)

#         display_grid = grid.copy()
#         if path1 != None:
#             for (y, x) in path1:
#                 display_grid[y][x] = 2   # mark path cells

#         if path2 != None:
#             for (y, x) in path2:
#                 display_grid[y][x] = 3   # mark path cells

#         if path3 != None:
#             for (y, x) in path3:
#                 display_grid[y][x] = 4   # mark path cells

#         plt.figure(figsize=(10,6))
#         plt.imshow(display_grid, cmap="viridis", interpolation="nearest")
#         plt.title("A* Path on Occupancy Grid")
#         plt.show()

#         # Print state for debugging / sending to bots
#         # for mid, state in bot_states.items():
#         #     print(f"  ID {mid}: center={state['center']}, heading={state['heading']:.1f}°")

#         cv2.imshow("Arena Camera - ArUco Tracking", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break


cv2.destroyAllWindows()


while True:
    ret, frame = setup_camera()
    if not ret:
        break

    frame, bot_states, grid, path1, path2, path3 = process_frame(frame)

    display_grid = grid.copy()
    if path1 != None:
        for (y, x) in path1:
            display_grid[y][x] = 2   # mark path cells

    if path2 != None:
        for (y, x) in path2:
            display_grid[y][x] = 3   # mark path cells

    if path3 != None:
        for (y, x) in path3:
            display_grid[y][x] = 4   # mark path cells

    plt.figure(figsize=(10,6))
    plt.imshow(display_grid, cmap="viridis", interpolation="nearest")
    plt.title("A* Path on Occupancy Grid")
    plt.show()

    # Print state for debugging / sending to bots
    # for mid, state in bot_states.items():
    #     print(f"  ID {mid}: center={state['center']}, heading={state['heading']:.1f}°")

    cv2.imshow("Arena Camera - ArUco Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()