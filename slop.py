import cv2
import numpy as np
from astar import astar, smooth_path
import time
import matplotlib.pyplot as plt
import socket
import math
import threading

# ═══════════════════════════════════════════════════
#              CONFIGURATION
# ═══════════════════════════════════════════════════
ROBOT_HOST = '192.168.0.120'
ROBOT_PORT = 80

# Arena physical size
ARENA_WIDTH_MM = 1750

# Grid resolution
CHUNK = 5

# Marker IDs
BOT_ID  = 9
GOAL_ID = 19

# Distance calibration
DISTANCE_SCALE = 1.0

# How close to goal before we stop (in grid cells)
GOAL_REACHED_CELLS = 4

# Red detection thresholds
RED_LOWER1 = np.array([0,   171, 98])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([170, 171, 98])
RED_UPPER2 = np.array([180, 255, 255])

# Recalculated at runtime
MM_PER_PIXEL = None
MM_PER_CELL  = None


# ═══════════════════════════════════════════════════
#              ROBOT WiFi CONNECTION
# ═══════════════════════════════════════════════════

def connect_robot(host, port, timeout=20):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((host, port))
    print(f"Connected to robot at {host}:{port}")
    return sock

def recv_line(sock):
    data = b''
    while True:
        byte = sock.recv(1)
        if not byte:
            raise ConnectionError("Socket closed by robot")
        if byte == b'\n':
            return data.decode().strip()
        data += byte

def send_command(sock, cmd, value):
    packet = f"{cmd}{value}\n"
    print(f"  -> Sending: {packet.strip()} ... ", end="", flush=True)
    sock.sendall(packet.encode())
    response = recv_line(sock)
    print(f"Robot: {response}")
    return response

def send_next_command(sock, commands):
    """Send only the first command from the list. Returns True if sent, False if empty."""
    if not commands:
        return False

    cmd_type, value = commands[0]

    if cmd_type == 'MOVE':
        robot_cmd = 'F'
        int_val = round(abs(value))
    elif cmd_type == 'ROTATE':
        robot_cmd = 'R' if value > 0 else 'L'
        int_val = round(abs(value))
    else:
        return False

    if int_val == 0:
        return False

    try:
        response = send_command(sock, robot_cmd, int_val)
        return response == "OK"
    except socket.timeout:
        print("TIMEOUT")
        return False


# ═══════════════════════════════════════════════════
#              ARUCO / VISION
# ═══════════════════════════════════════════════════
DICT_TYPE = cv2.aruco.DICT_4X4_50
arucoDict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
arucoParams = cv2.aruco.DetectorParameters()
arucoParams.adaptiveThreshWinSizeMin = 3
arucoParams.adaptiveThreshWinSizeMax = 23
arucoParams.adaptiveThreshWinSizeStep = 10
arucoParams.adaptiveThreshConstant = 7
arucoParams.minMarkerPerimeterRate = 0.02
arucoParams.maxMarkerPerimeterRate = 4.0
arucoParams.polygonalApproxAccuracyRate = 0.03
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)


def get_marker_center(corners_single):
    pts = corners_single[0]
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    return cx, cy

def get_marker_heading(corners_single):
    pts = corners_single[0]
    forward_mid = ((pts[1] + pts[2]) / 2)
    center = np.mean(pts, axis=0)
    delta = forward_mid - center
    angle = np.degrees(np.arctan2(-delta[1], delta[0]))
    return angle

def resize_to_fit(image, max_width=1280, max_height=720):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def detect_markers(frame):
    """Detect markers without drawing. Returns bot_states dict."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    bot_states = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            cx, cy = get_marker_center(corners[i])
            heading = get_marker_heading(corners[i])
            bot_states[marker_id] = {"center": (cx, cy), "heading": heading}
    return bot_states

def detect_and_draw_markers(frame):
    """Detect markers and draw overlays. Returns bot_states and annotated frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    bot_states = {}
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, marker_id in enumerate(ids.flatten()):
            cx, cy = get_marker_center(corners[i])
            heading = get_marker_heading(corners[i])
            bot_states[marker_id] = {"center": (cx, cy), "heading": heading}
            label = f"ID {marker_id} | {heading:.1f} deg"
            cv2.putText(frame, label, (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            arrow_len = 40
            end_x = int(cx + arrow_len * math.cos(math.radians(heading)))
            end_y = int(cy - arrow_len * math.sin(math.radians(heading)))
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (255, 0, 0), 2, tipLength=0.3)
    return bot_states, frame

def get_red_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    red_mask = mask1 | mask2
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, morph_kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, morph_kernel)
    return red_mask


# ═══════════════════════════════════════════════════
#              PATH PROCESSING
# ═══════════════════════════════════════════════════

def get_next_commands(smoothed_path, current_heading_deg, max_commands=2):
    """
    Get only the next 1-2 commands from the path.
    In continuous mode we only need the immediate next move,
    since we'll recalculate after it completes.
    """
    if not smoothed_path or len(smoothed_path) < 2:
        return []

    commands = []
    heading = current_heading_deg

    # Only process the first segment
    r0, c0 = smoothed_path[0]
    r1, c1 = smoothed_path[1]

    dx_cells = c1 - c0
    dy_cells = r0 - r1

    dx_px = dx_cells * CHUNK
    dy_px = dy_cells * CHUNK

    target_angle = math.degrees(math.atan2(dy_px, dx_px))
    turn = (target_angle - heading + 180) % 360 - 180

    dist_px = math.sqrt(dx_px**2 + dy_px**2)
    dist_mm = dist_px * MM_PER_PIXEL * DISTANCE_SCALE

    if abs(turn) > 1:
        commands.append(('ROTATE', round(turn, 1)))

    if dist_mm > 1:
        commands.append(('MOVE', round(dist_mm, 1)))

    return commands


def distance_to_goal(bot_states):
    """Distance between bot and goal in grid cells. Returns None if either not detected."""
    if BOT_ID not in bot_states or GOAL_ID not in bot_states:
        return None
    bx, by = bot_states[BOT_ID]["center"]
    gx, gy = bot_states[GOAL_ID]["center"]
    dx = (bx - gx) / CHUNK
    dy = (by - gy) / CHUNK
    return math.sqrt(dx**2 + dy**2)


# ═══════════════════════════════════════════════════
#              FAST FRAME PROCESSING (no GUI waits)
# ═══════════════════════════════════════════════════

def compute_path(frame):
    """
    Process frame and return path. No cv2.waitKey or imshow pauses.
    Returns (bot_states, path, inflated_grid) or (bot_states, [], None) on failure.
    """
    bot_states = detect_markers(frame)

    if BOT_ID not in bot_states or GOAL_ID not in bot_states:
        return bot_states, [], None

    marker_centers = [s["center"] for s in bot_states.values()]

    red_mask = get_red_mask(frame)
    edges = cv2.Canny(red_mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    HEIGHT, WIDTH = edges.shape
    CHUNK_H = HEIGHT // CHUNK
    CHUNK_W = WIDTH // CHUNK

    grid = np.zeros((CHUNK_H, CHUNK_W), dtype=int)
    mask = np.zeros_like(edges)

    for cnt in contours:
        cv2.drawContours(mask, [cnt], -1, 255, 1)

    EXCLUDE_RADIUS = 60
    for (mx, my) in marker_centers:
        cv2.circle(mask, (mx, my), EXCLUDE_RADIUS, 0, -1)

    ys, xs = np.where(mask == 255)
    for x, y in zip(xs, ys):
        cx_g = min(x // CHUNK, CHUNK_W - 1)
        cy_g = min(y // CHUNK, CHUNK_H - 1)
        grid[cy_g, cx_g] = 1

    inflate_radius = 3
    grid_img = (grid.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflate_radius*2+1, inflate_radius*2+1))
    inflated = cv2.dilate(grid_img, kernel)
    inflated_grid = (inflated > 0).astype(int)

    start = (bot_states[BOT_ID]["center"][1]//CHUNK, bot_states[BOT_ID]["center"][0]//CHUNK)
    goal  = (bot_states[GOAL_ID]["center"][1]//CHUNK, bot_states[GOAL_ID]["center"][0]//CHUNK)

    start = (max(0, min(start[0], CHUNK_H-1)), max(0, min(start[1], CHUNK_W-1)))
    goal  = (max(0, min(goal[0], CHUNK_H-1)), max(0, min(goal[1], CHUNK_W-1)))

    clear_cells(inflated_grid, start, radius=3)
    clear_cells(inflated_grid, goal, radius=3)

    raw_path = astar(inflated_grid, start, goal)

    if raw_path:
        path = smooth_path(inflated_grid, raw_path)
        return bot_states, path, inflated_grid
    else:
        return bot_states, [], inflated_grid


def clear_cells(grid, pos, radius=2):
    rows, cols = grid.shape
    r, c = pos
    for rr in range(max(0, r - radius), min(rows, r + radius + 1)):
        for cc in range(max(0, c - radius), min(cols, c + radius + 1)):
            grid[rr, cc] = 0


# ═══════════════════════════════════════════════════
#              FULL PROCESS (for manual SPACE mode)
# ═══════════════════════════════════════════════════

def process_frame_full(frame):
    """Full processing with GUI display — used in manual SPACE mode."""
    grid = np.zeros((1, 1), dtype=int)
    inflated_grid = grid
    path = []

    bot_states, frame = detect_and_draw_markers(frame)

    if not bot_states:
        print("WARNING: No markers detected!")
        return frame, bot_states, grid, inflated_grid, path

    if GOAL_ID not in bot_states or BOT_ID not in bot_states:
        print(f"WARNING: Missing bot or goal marker!")
        return frame, bot_states, grid, inflated_grid, path

    marker_centers = [s["center"] for s in bot_states.values()]
    red_mask = get_red_mask(frame)
    edges = cv2.Canny(red_mask, 50, 150)

    cv2.imshow("EDGE MASK", resize_to_fit(edges))
    cv2.waitKey(0)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    HEIGHT, WIDTH = edges.shape
    CHUNK_H = HEIGHT // CHUNK
    CHUNK_W = WIDTH // CHUNK
    grid = np.zeros((CHUNK_H, CHUNK_W), dtype=int)
    mask = np.zeros_like(edges)

    cv2.imshow("RED MASK", resize_to_fit(red_mask))
    cv2.waitKey(0)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(frame, [approx], -1, (0, 0, 255), 5)
        cv2.drawContours(mask, [cnt], -1, 255, 1)

    EXCLUDE_RADIUS = 60
    for (mx, my) in marker_centers:
        cv2.circle(mask, (mx, my), EXCLUDE_RADIUS, 0, -1)

    ys, xs = np.where(mask == 255)
    for x, y in zip(xs, ys):
        cx_g = min(x // CHUNK, CHUNK_W - 1)
        cy_g = min(y // CHUNK, CHUNK_H - 1)
        grid[cy_g, cx_g] = 1

    inflate_radius = 3
    grid_img = (grid.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflate_radius*2+1, inflate_radius*2+1))
    inflated = cv2.dilate(grid_img, kernel)
    inflated_grid = (inflated > 0).astype(int)

    start = (bot_states[BOT_ID]["center"][1]//CHUNK, bot_states[BOT_ID]["center"][0]//CHUNK)
    goal  = (bot_states[GOAL_ID]["center"][1]//CHUNK, bot_states[GOAL_ID]["center"][0]//CHUNK)

    clear_cells(inflated_grid, start, radius=3)
    clear_cells(inflated_grid, goal, radius=3)

    raw_path = astar(inflated_grid, start, goal)
    if raw_path:
        path = smooth_path(inflated_grid, raw_path)
        print(f"Bot {BOT_ID}: {len(raw_path)} A* -> {len(path)} smoothed")

    return frame, bot_states, grid, inflated_grid, path


def draw_smooth_path_on_grid(display_grid, path, value):
    if not path or len(path) < 2:
        return
    rows, cols = display_grid.shape
    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        steps = max(abs(r1 - r0), abs(c1 - c0))
        if steps == 0:
            continue
        for t in range(steps + 1):
            r = int(round(r0 + t * (r1 - r0) / steps))
            c = int(round(c0 + t * (c1 - c0) / steps))
            if 0 <= r < rows and 0 <= c < cols:
                display_grid[r][c] = value


# ═══════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════

if __name__ == "__main__":

    robot_sock = None
    try:
        robot_sock = connect_robot(ROBOT_HOST, ROBOT_PORT)
    except Exception as e:
        print(f"Could not connect to robot: {e}")
        print("Running in vision-only mode")

    cap = cv2.VideoCapture(1)

    with np.load('camera_params.npz') as data:
        mtx, dist_coeffs = data['mtx'], data['dist']
        dist_coeffs = dist_coeffs * 0.1

    first_frame = True
    newcameramtx, roi = None, None
    continuous_mode = False  # toggled with 'C'

    print(f"Bot: ID {BOT_ID}  Goal: ID {GOAL_ID}")
    print(f"DISTANCE_SCALE: {DISTANCE_SCALE}")
    print()
    print("Controls:")
    print("  SPACE = single-shot: capture + full pathfind + send all commands")
    print("  C     = continuous mode: replan + send next command every cycle")
    print("  S     = emergency stop + exit continuous mode")
    print("  R     = preview red detection")
    print("  Q     = quit")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if first_frame:
            h, w = frame.shape[:2]
            print(f"Raw camera resolution: {w}x{h}")
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist_coeffs, (w, h), 1, (w, h))
            first_frame = False

        undistorted = cv2.undistort(frame, mtx, dist_coeffs, None, newcameramtx)
        x, y, rw, rh = roi
        undistorted = undistorted[y:y+rh, x:x+rw]

        if MM_PER_PIXEL is None:
            actual_h, actual_w = undistorted.shape[:2]
            MM_PER_PIXEL = ARENA_WIDTH_MM / actual_w
            MM_PER_CELL = MM_PER_PIXEL * CHUNK
            print(f"Undistorted: {actual_w}x{actual_h}")
            print(f"Scale: {MM_PER_PIXEL:.4f} mm/px, {MM_PER_CELL:.2f} mm/cell")
            print()

        # ─── Live display ───
        live_display = undistorted.copy()
        live_states, live_display = detect_and_draw_markers(live_display)

        n_markers = len(live_states)
        marker_color = (0, 255, 0) if n_markers >= 2 else (0, 0, 255)

        status = f"Markers: {n_markers}"
        if BOT_ID in live_states:
            status += f" | Bot: {live_states[BOT_ID]['heading']:.0f} deg"
        if GOAL_ID in live_states:
            dist = distance_to_goal(live_states)
            if dist is not None:
                status += f" | Goal: {dist:.0f} cells"

        mode_label = "CONTINUOUS" if continuous_mode else "MANUAL"
        mode_color = (0, 165, 255) if continuous_mode else (200, 200, 200)
        cv2.putText(live_display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, marker_color, 2)
        cv2.putText(live_display, f"Mode: {mode_label}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

        cv2.imshow("Live Webcam Feed", resize_to_fit(live_display))

        # ─── Continuous feedback loop ───
        if continuous_mode and robot_sock:
            # Check if we've reached the goal
            dist = distance_to_goal(live_states)
            if dist is not None and dist < GOAL_REACHED_CELLS:
                print(f"\n=== GOAL REACHED! Distance: {dist:.1f} cells ===\n")
                try:
                    send_command(robot_sock, 'S', 0)
                except:
                    pass
                continuous_mode = False
            elif BOT_ID in live_states and GOAL_ID in live_states:
                # Recompute path from current position
                bot_states, path, _ = compute_path(undistorted.copy())

                if path and len(path) >= 2:
                    heading = bot_states[BOT_ID]["heading"]
                    commands = get_next_commands(path, heading)

                    if commands:
                        print(f"[CONTINUOUS] dist={dist:.0f} cells | "
                              f"heading={heading:.0f} | cmd={commands[0]}")

                        # Send each command and wait for OK
                        for cmd_type, value in commands:
                            if cmd_type == 'MOVE':
                                robot_cmd = 'F'
                                int_val = round(abs(value))
                            elif cmd_type == 'ROTATE':
                                robot_cmd = 'R' if value > 0 else 'L'
                                int_val = round(abs(value))
                            else:
                                continue
                            if int_val > 0:
                                try:
                                    send_command(robot_sock, robot_cmd, int_val)
                                except socket.timeout:
                                    print("TIMEOUT — stopping continuous mode")
                                    continuous_mode = False
                                    break
                else:
                    print("[CONTINUOUS] No path found, retrying...")
            else:
                print("[CONTINUOUS] Markers not detected, waiting...")

        # ─── Key handling ───
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('s'):
            continuous_mode = False
            if robot_sock:
                try:
                    send_command(robot_sock, 'S', 0)
                except:
                    pass
            print("EMERGENCY STOP — continuous mode OFF")

        if key == ord('c'):
            if robot_sock:
                continuous_mode = not continuous_mode
                if continuous_mode:
                    print("\n=== CONTINUOUS MODE ON ===")
                    print("Robot will replan and move every cycle.")
                    print("Press S to stop, Q to quit.\n")
                else:
                    print("\n=== CONTINUOUS MODE OFF ===\n")
                    try:
                        send_command(robot_sock, 'S', 0)
                    except:
                        pass
            else:
                print("No robot connection — cannot start continuous mode")

        if key == ord('r'):
            red_preview = get_red_mask(undistorted.copy())
            red_bgr = cv2.cvtColor(red_preview, cv2.COLOR_GRAY2BGR)
            side_by_side = np.hstack([
                resize_to_fit(undistorted, max_width=640, max_height=360),
                resize_to_fit(red_bgr, max_width=640, max_height=360)
            ])
            cv2.imshow("Red Detection Preview", side_by_side)

        if key == ord(' '):
            print("\n========== SINGLE SHOT ==========")
            continuous_mode = False

            proc_frame, bot_states, grid, inflated_grid, path = process_frame_full(
                undistorted.copy()
            )

            display_grid = grid.copy().astype(float)
            draw_smooth_path_on_grid(display_grid, path, 2)

            cv2.imshow("Processed Frame", resize_to_fit(proc_frame))

            plt.figure(figsize=(10, 6))
            plt.imshow(display_grid, cmap="viridis", interpolation="nearest")
            plt.title("Smoothed A* Path")
            plt.show()

            if path and BOT_ID in bot_states:
                current_heading = bot_states[BOT_ID]["heading"]

                # In single-shot, send ALL commands
                all_commands = []
                heading = current_heading
                for i in range(1, len(path)):
                    r0, c0 = path[i-1]
                    r1, c1 = path[i]
                    dx_px = (c1 - c0) * CHUNK
                    dy_px = (r0 - r1) * CHUNK
                    target = math.degrees(math.atan2(dy_px, dx_px))
                    turn = (target - heading + 180) % 360 - 180
                    dist_mm = math.sqrt(dx_px**2 + dy_px**2) * MM_PER_PIXEL * DISTANCE_SCALE
                    if abs(turn) > 1:
                        all_commands.append(('ROTATE', round(turn, 1)))
                    if dist_mm > 1:
                        all_commands.append(('MOVE', round(dist_mm, 1)))
                    heading = target

                print(f"Commands ({len(all_commands)}):")
                for cmd in all_commands:
                    print(f"  {cmd}")

                if robot_sock:
                    print(f"\n--- Executing {len(all_commands)} commands ---")
                    for i, (ct, v) in enumerate(all_commands):
                        print(f"[{i+1}/{len(all_commands)}] ", end="")
                        if ct == 'MOVE':
                            send_command(robot_sock, 'F', round(abs(v)))
                        elif ct == 'ROTATE':
                            rc = 'R' if v > 0 else 'L'
                            send_command(robot_sock, rc, round(abs(v)))
                    print("--- Done ---\n")

    cap.release()
    cv2.destroyAllWindows()
    if robot_sock:
        try:
            send_command(robot_sock, 'S', 0)
        except:
            pass
        robot_sock.close()
        print("Robot disconnected")
