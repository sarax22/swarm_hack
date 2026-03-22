import cv2
import numpy as np
from astar import astar
import time
import socket
import math

# ═══════════════════════════════════════════════════
#                   CONFIG
# ═══════════════════════════════════════════════════

# Robot IPs — update for your MONAs
BOTS = {
    'rear':  {'marker_id': 9, 'ip': '192.168.0.120'},
    # 'left':  {'marker_id': 2, 'ip': '192.168.0.121'},
    # 'right': {'marker_id': 3, 'ip': '192.168.0.122'},
}
ROBOT_PORT = 80

# Marker IDs
TROLLEY_ID = 19
GOAL_MARKER_A = 19
GOAL_MARKER_B = 19

# Arena scale
MM_PER_PIXEL = 1750 / 1920
CHUNK = 5

# Navigation tuning
WAYPOINT_THRESHOLD_PX = 30
ANGLE_THRESHOLD_DEG = 10
FORMATION_OFFSET_PX = 80
REPLAN_INTERVAL = 3.0
PUSH_ARRIVED_THRESHOLD_PX = 40


# ═══════════════════════════════════════════════════
#              ROBOT WiFi CONNECTION
# ═══════════════════════════════════════════════════

def connect_robot(host, port=80, timeout=20):
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


def send_move_command(sock, cmd, value):
    packet = f"{cmd}{value}\n"
    sock.sendall(packet.encode())
    response = recv_line(sock)
    return response


def send_stop(sock):
    try:
        send_move_command(sock, 'S', 0)
    except:
        pass


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


def detect_markers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    bot_states = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            pts = corners[i][0]
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            top_mid = (pts[0] + pts[1]) / 2
            center = np.mean(pts, axis=0)
            delta = top_mid - center
            heading = np.degrees(np.arctan2(-delta[1], delta[0]))
            bot_states[marker_id] = {"center": (cx, cy), "heading": heading}
    return bot_states


def build_occupancy_grid(frame, bot_states):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    red_mask = mask1 | mask2

    edges = cv2.Canny(red_mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    HEIGHT, WIDTH = edges.shape
    CHUNK_H = HEIGHT // CHUNK
    CHUNK_W = WIDTH // CHUNK
    grid = np.zeros((CHUNK_H, CHUNK_W), dtype=int)
    mask = np.zeros_like(edges)

    for cnt in contours:
        cv2.drawContours(mask, [cnt], -1, 255, 1)

    # Exclude all markers from obstacle detection
    for state in bot_states.values():
        mx, my = state["center"]
        cv2.circle(mask, (mx, my), 60, 0, -1)

    ys, xs = np.where(mask == 255)
    for x, y in zip(xs, ys):
        gc = min(x // CHUNK, CHUNK_W - 1)
        gr = min(y // CHUNK, CHUNK_H - 1)
        grid[gr, gc] = 1

    # Add trolley as obstacle
    if TROLLEY_ID in bot_states:
        tx, ty = bot_states[TROLLEY_ID]["center"]
        tgc, tgr = tx // CHUNK, ty // CHUNK
        rows, cols = grid.shape
        for r in range(max(0, tgr - 6), min(rows, tgr + 7)):
            for c in range(max(0, tgc - 6), min(cols, tgc + 7)):
                if (r - tgr) ** 2 + (c - tgc) ** 2 <= 36:
                    grid[r, c] = 1

    # Inflate
    grid_img = (grid.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    inflated = cv2.dilate(grid_img, kernel)
    return (inflated > 0).astype(int)


# ═══════════════════════════════════════════════════
#              PATH PLANNING
# ═══════════════════════════════════════════════════

def simplify_path(path):
    if not path or len(path) < 3:
        return path
    simplified = [path[0]]
    for i in range(1, len(path) - 1):
        dy1 = path[i][0] - path[i - 1][0]
        dx1 = path[i][1] - path[i - 1][1]
        dy2 = path[i + 1][0] - path[i][0]
        dx2 = path[i + 1][1] - path[i][1]
        if (dx1, dy1) != (dx2, dy2):
            simplified.append(path[i])
    simplified.append(path[-1])
    return simplified


def clear_cells(grid, goal, radius=3):
    rows, cols = grid.shape
    gr, gc = goal
    for r in range(max(0, gr - radius), min(rows, gr + radius + 1)):
        for c in range(max(0, gc - radius), min(cols, gc + radius + 1)):
            grid[r, c] = 0


def get_formation_goals(bot_states):
    if TROLLEY_ID not in bot_states:
        return None
    tx, ty = bot_states[TROLLEY_ID]["center"]
    h = math.radians(bot_states[TROLLEY_ID]["heading"])
    off = FORMATION_OFFSET_PX

    goals = {
        'rear':  (int(ty + off * math.sin(h)) // CHUNK,
                  int(tx - off * math.cos(h)) // CHUNK),
        'left':  (int(ty - off * math.cos(h)) // CHUNK,
                  int(tx - off * math.sin(h)) // CHUNK),
        'right': (int(ty + off * math.cos(h)) // CHUNK,
                  int(tx + off * math.sin(h)) // CHUNK),
    }
    return goals


def plan_approach(bot_states, grid):
    goals = get_formation_goals(bot_states)
    if goals is None:
        return {}

    paths = {}
    for role, info in BOTS.items():
        mid = info['marker_id']
        if mid not in bot_states or role not in goals:
            continue

        goal = goals[role]
        clear_cells(grid, goal, radius=3)

        bx, by = bot_states[mid]["center"]
        start = (by // CHUNK, bx // CHUNK)

        raw = astar(grid, start, goal)
        if raw:
            paths[role] = simplify_path(raw)
        else:
            print(f"  No path for {role} bot")
    return paths


def get_goal_px(bot_states):
    if GOAL_MARKER_A not in bot_states or GOAL_MARKER_B not in bot_states:
        return None
    ax, ay = bot_states[GOAL_MARKER_A]["center"]
    bx, by = bot_states[GOAL_MARKER_B]["center"]
    return ((ax + bx) // 2, (ay + by) // 2)


def offset_path(path, side, offset_px=None):
    if offset_px is None:
        offset_px = FORMATION_OFFSET_PX
    if not path or len(path) < 2:
        return path

    off_cells = offset_px / CHUNK
    result = []
    for i in range(len(path)):
        r, c = path[i]
        if i < len(path) - 1:
            dr = path[i + 1][0] - r
            dc = path[i + 1][1] - c
        else:
            dr = r - path[i - 1][0]
            dc = c - path[i - 1][1]
        length = math.sqrt(dr ** 2 + dc ** 2)
        if length == 0:
            result.append((r, c))
            continue
        dr /= length
        dc /= length

        if side == 'rear':
            nr = r - dr * off_cells
            nc = c - dc * off_cells
        elif side == 'left':
            nr = r - dc * off_cells
            nc = c + dr * off_cells
        elif side == 'right':
            nr = r + dc * off_cells
            nc = c - dr * off_cells
        else:
            nr, nc = r, c
        result.append((round(nr), round(nc)))
    return result


def plan_push(bot_states, grid):
    if TROLLEY_ID not in bot_states:
        return {}
    goal_pos = get_goal_px(bot_states)
    if goal_pos is None:
        return {}

    tx, ty = bot_states[TROLLEY_ID]["center"]
    start = (ty // CHUNK, tx // CHUNK)
    goal = (goal_pos[1] // CHUNK, goal_pos[0] // CHUNK)

    # Clear trolley and goal areas so path can be found
    clear_cells(grid, start, radius=8)
    clear_cells(grid, goal, radius=4)

    raw = astar(grid, start, goal)
    if raw is None:
        print("No push path found!")
        return {}

    trolley_path = simplify_path(raw)
    paths = {}
    for role in BOTS:
        paths[role] = offset_path(trolley_path, side=role)
    return paths


# ═══════════════════════════════════════════════════
#         CLOSED-LOOP SINGLE-STEP NAVIGATION
# ═══════════════════════════════════════════════════

def wp_to_px(path, idx):
    if idx >= len(path):
        return None
    r, c = path[idx]
    return (c * CHUNK + CHUNK // 2, r * CHUNK + CHUNK // 2)


def navigate_step(sock, bot_states, marker_id, target_px, max_fwd=60):
    """
    Send one corrective command. Returns True if target reached.
    """
    if marker_id not in bot_states:
        send_stop(sock)
        return False

    bx, by = bot_states[marker_id]["center"]
    heading = bot_states[marker_id]["heading"]
    tx, ty = target_px

    dx = tx - bx
    dy = -(ty - by)
    dist = math.sqrt(dx ** 2 + dy ** 2)

    if dist < WAYPOINT_THRESHOLD_PX:
        return True

    target_angle = math.degrees(math.atan2(dy, dx))
    turn = (target_angle - heading + 180) % 360 - 180

    try:
        if abs(turn) > ANGLE_THRESHOLD_DEG:
            cmd = 'R' if turn > 0 else 'L'
            val = min(round(abs(turn)), 45)
            send_move_command(sock, cmd, val)
        else:
            burst = min(round(dist * MM_PER_PIXEL), max_fwd)
            burst = max(burst, 10)
            send_move_command(sock, 'F', burst)
    except Exception as e:
        print(f"    Command failed: {e}")

    return False


def all_arrived(bot_states, paths):
    for role, info in BOTS.items():
        mid = info['marker_id']
        if mid not in bot_states:
            return False
        if role not in paths or not paths[role]:
            return False
        final = wp_to_px(paths[role], len(paths[role]) - 1)
        if final is None:
            return False
        bx, by = bot_states[mid]["center"]
        if math.sqrt((final[0] - bx) ** 2 + (final[1] - by) ** 2) > WAYPOINT_THRESHOLD_PX:
            return False
    return True


def trolley_at_goal(bot_states):
    if TROLLEY_ID not in bot_states:
        return False
    goal = get_goal_px(bot_states)
    if goal is None:
        return False
    tx, ty = bot_states[TROLLEY_ID]["center"]
    return math.sqrt((goal[0] - tx) ** 2 + (goal[1] - ty) ** 2) < PUSH_ARRIVED_THRESHOLD_PX


# ═══════════════════════════════════════════════════
#                    MAIN LOOP
# ═══════════════════════════════════════════════════

STATE_IDLE = 'IDLE'
STATE_APPROACH = 'APPROACH'
STATE_PUSH = 'PUSH'
STATE_DONE = 'DONE'

cv2.namedWindow("HSV Tuning")
cv2.createTrackbar("H min", "HSV Tuning", 0, 180, lambda x: None)
cv2.createTrackbar("H max", "HSV Tuning", 10, 180, lambda x: None)
cv2.createTrackbar("S min", "HSV Tuning", 100, 255, lambda x: None)
cv2.createTrackbar("S max", "HSV Tuning", 255, 255, lambda x: None)
cv2.createTrackbar("V min", "HSV Tuning", 100, 255, lambda x: None)
cv2.createTrackbar("V max", "HSV Tuning", 255, 255, lambda x: None)
cv2.createTrackbar("H min 2", "HSV Tuning", 170, 180, lambda x: None)
cv2.createTrackbar("H max 2", "HSV Tuning", 180, 180, lambda x: None)

if __name__ == "__main__":

    # --- Connect robots ---
    robot_socks = {}
    for role, info in BOTS.items():
        try:
            robot_socks[role] = connect_robot(info['ip'], ROBOT_PORT)
        except Exception as e:
            print(f"Could not connect {role} bot ({info['ip']}): {e}")

    if not robot_socks:
        print("No robots connected — vision-only mode.")

    # --- Camera ---
    cap = cv2.VideoCapture(1)
    with np.load('camera_params.npz') as data:
        mtx, dist_coeffs = data['mtx'], data['dist']
        dist_coeffs = dist_coeffs * 0.6

    cam_h, cam_w = None, None
    newcameramtx, roi_rect = None, None

    # --- State ---
    state = STATE_IDLE
    paths = {}
    wp_idx = {}          # role -> current waypoint index
    last_replan = 0

    print("\nSPACE = start | R = reset | Q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort
        if cam_h is None:
            cam_h, cam_w = frame.shape[:2]
            newcameramtx, roi_rect = cv2.getOptimalNewCameraMatrix(
                mtx, dist_coeffs, (cam_w, cam_h), 1, (cam_w, cam_h))

        undistorted = cv2.undistort(frame, mtx, dist_coeffs, None, newcameramtx)
        rx, ry, rw, rh = roi_rect
        undistorted = undistorted[ry:ry + rh, rx:rx + rw]

        # --- Detect every frame ---
        bot_states = detect_markers(undistorted)

        # --- Draw overlays ---
        display = undistorted.copy()
        for mid, s in bot_states.items():
            cx, cy = s["center"]
            cv2.circle(display, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(display, f"{mid}|{s['heading']:.0f}",
                        (cx + 8, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            

        # --- Live HSV tuning ---
        hsv_frame = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HSV)

        h1 = cv2.getTrackbarPos("H min", "HSV Tuning")
        h2 = cv2.getTrackbarPos("H max", "HSV Tuning")
        s1 = cv2.getTrackbarPos("S min", "HSV Tuning")
        s2 = cv2.getTrackbarPos("S max", "HSV Tuning")
        v1 = cv2.getTrackbarPos("V min", "HSV Tuning")
        v2 = cv2.getTrackbarPos("V max", "HSV Tuning")
        h1b = cv2.getTrackbarPos("H min 2", "HSV Tuning")
        h2b = cv2.getTrackbarPos("H max 2", "HSV Tuning")

        mask_a = cv2.inRange(hsv_frame, np.array([h1, s1, v1]), np.array([h2, s2, v2]))
        mask_b = cv2.inRange(hsv_frame, np.array([h1b, s1, v1]), np.array([h2b, s2, v2]))
        combined = mask_a | mask_b

        # Show mask and overlay
        mask_colored = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(undistorted, 0.7, mask_colored, 0.3, 0)

        cv2.imshow("HSV Tuning", np.vstack([overlay, mask_colored]))



        goal_pos = get_goal_px(bot_states)
        if goal_pos:
            cv2.circle(display, goal_pos, 10, (0, 0, 255), 2)
            cv2.putText(display, "GOAL", (goal_pos[0] + 12, goal_pos[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        cv2.putText(display, f"State: {state}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        now = time.time()

        # ═══════════ APPROACH ═══════════
        if state == STATE_APPROACH:

            if now - last_replan > REPLAN_INTERVAL:
                grid = build_occupancy_grid(undistorted, bot_states)
                paths = plan_approach(bot_states, grid)
                wp_idx = {role: 0 for role in paths}
                last_replan = now
                print(f"[APPROACH] Replanned — {len(paths)} paths")

            for role, info in BOTS.items():
                if role not in paths or role not in robot_socks:
                    continue
                mid = info['marker_id']
                p = paths[role]
                idx = wp_idx.get(role, 0)
                if idx >= len(p):
                    continue

                target = wp_to_px(p, idx)
                if target:
                    cv2.circle(display, target, 6, (255, 0, 255), -1)
                    reached = navigate_step(robot_socks[role], bot_states, mid, target)
                    if reached:
                        wp_idx[role] = idx + 1
                        print(f"  [{role}] waypoint {idx + 1}/{len(p)}")

            if all_arrived(bot_states, paths):
                print("\n=== Formation reached — switching to PUSH ===\n")
                state = STATE_PUSH
                paths = {}
                wp_idx = {}
                last_replan = 0

        # ═══════════ PUSH ═══════════
        elif state == STATE_PUSH:

            if now - last_replan > REPLAN_INTERVAL:
                grid = build_occupancy_grid(undistorted, bot_states)
                # Remove trolley obstacle for push path planning
                if TROLLEY_ID in bot_states:
                    tx, ty = bot_states[TROLLEY_ID]["center"]
                    clear_cells(grid, (ty // CHUNK, tx // CHUNK), radius=8)
                paths = plan_push(bot_states, grid)
                wp_idx = {role: 0 for role in paths}
                last_replan = now
                print(f"[PUSH] Replanned — {len(paths)} paths")

            for role, info in BOTS.items():
                if role not in paths or role not in robot_socks:
                    continue
                mid = info['marker_id']
                p = paths[role]
                idx = wp_idx.get(role, 0)
                if idx >= len(p):
                    continue

                target = wp_to_px(p, idx)
                if target:
                    cv2.circle(display, target, 6, (0, 255, 255), -1)
                    reached = navigate_step(
                        robot_socks[role], bot_states, mid, target, max_fwd=40)
                    if reached:
                        wp_idx[role] = idx + 1
                        print(f"  [{role}] push wp {idx + 1}/{len(p)}")

            if trolley_at_goal(bot_states):
                print("\n=== TROLLEY AT GOAL — RUN COMPLETE! ===\n")
                state = STATE_DONE
                for sock in robot_socks.values():
                    send_stop(sock)

        # ═══════════ DONE ═══════════
        elif state == STATE_DONE:
            cv2.putText(display, "COMPLETE!", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # ═══════════ DISPLAY ═══════════
        cv2.imshow("Arena", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and state == STATE_IDLE:
            print("\n=== GO — APPROACH phase ===\n")
            state = STATE_APPROACH
            last_replan = 0
            wp_idx = {}
        elif key == ord('r'):
            print("\n=== RESET ===\n")
            state = STATE_IDLE
            paths = {}
            wp_idx = {}
            for sock in robot_socks.values():
                send_stop(sock)

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    for sock in robot_socks.values():
        send_stop(sock)
        sock.close()
    print("Done.")