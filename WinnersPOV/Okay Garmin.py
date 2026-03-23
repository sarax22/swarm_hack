import cv2
import numpy as np
from astar import astar, smooth_path
import time
import socket
import math
import threading

# ═══════════════════════════════════════════════════
#              CONFIGURATION
# ═══════════════════════════════════════════════════
ROBOT_HOST = 'IP address from esp'
ROBOT_PORT = 80

ARENA_WIDTH_MM = 1750
CHUNK = 5

BOT_ID      = 9
TROLLEY_ID  = 6
GOAL_A_ID   = 0
GOAL_B_ID   = 11

DISTANCE_SCALE = 1.0

BOT_DIAMETER_MM = 80
BOT_RADIUS_MM   = BOT_DIAMETER_MM / 2

TROLLEY_RADIUS_MM = 40
TROLLEY_RADIUS_PX = None

LINEUP_OFFSET_MM = 120
LINEUP_OFFSET_PX = None

BEHIND_TOLERANCE_CELLS = 10
PUSH_GOAL_REACHED_CELLS = 10
ANGLE_TOLERANCE_DEG = 25
PUSH_STEP_MM = 80
BACKUP_MM = 60  # how far to reverse when stuck or off-center

MAX_BLIND_PUSHES = 5

# Stuck detection: if trolley hasn't moved this many cells in N pushes, we're stuck
STUCK_PUSH_COUNT = 3
STUCK_MOVE_THRESHOLD_CELLS = 2  # trolley must move at least this much per N pushes

# Off-center threshold: if bot is this far from the push line, back up and realign
OFF_CENTER_THRESHOLD_CELLS = 8

LIVE_DIM = 0.9

RED_LOWER1 = np.array([0, 74, 0])
RED_UPPER1 = np.array([6, 255, 255])
RED_LOWER2 = np.array([80, 74, 0])
RED_UPPER2 = np.array([180, 255, 255])

MM_PER_PIXEL = None
MM_PER_CELL  = None
BOT_INFLATE_CELLS = None


# ═══════════════════════════════════════════════════
#              ROBOT WiFi
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
            raise ConnectionError("Socket closed")
        if byte == b'\n':
            return data.decode().strip()
        data += byte

def send_command(sock, cmd, value):
    packet = f"{cmd}{value}\n"
    print(f"  -> {packet.strip()} ... ", end="", flush=True)
    sock.sendall(packet.encode())
    response = recv_line(sock)
    print(f"{response}")
    return response

def send_move(sock, cmd_type, value):
    if cmd_type == 'MOVE':
        robot_cmd = 'F'
        int_val = round(abs(value))
    elif cmd_type == 'ROTATE':
        robot_cmd = 'R' if value > 0 else 'L'
        int_val = round(abs(value))
    else:
        return False
    if int_val == 0:
        return True
    try:
        response = send_command(sock, robot_cmd, int_val)
        return response == "OK"
    except socket.timeout:
        print("TIMEOUT")
        return False


# ═══════════════════════════════════════════════════
#              ARUCO
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
    return int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))

def get_marker_heading(corners_single):
    pts = corners_single[0]
    forward_mid = ((pts[1] + pts[2]) / 2)
    center = np.mean(pts, axis=0)
    delta = forward_mid - center
    return np.degrees(np.arctan2(-delta[1], delta[0]))

def resize_to_fit(image, max_width=1280, max_height=720):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        return cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return image

def detect_markers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    states = {}
    if ids is not None:
        for i, mid in enumerate(ids.flatten()):
            cx, cy = get_marker_center(corners[i])
            states[mid] = {"center": (cx, cy), "heading": get_marker_heading(corners[i])}
    return states

def detect_and_draw_markers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    states = {}
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, mid in enumerate(ids.flatten()):
            cx, cy = get_marker_center(corners[i])
            h = get_marker_heading(corners[i])
            states[mid] = {"center": (cx, cy), "heading": h}
            cv2.putText(frame, f"ID{mid}|{h:.0f}", (cx+10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            al = 40
            cv2.arrowedLine(frame, (cx, cy),
                            (int(cx + al * math.cos(math.radians(h))),
                             int(cy - al * math.sin(math.radians(h)))),
                            (255, 0, 0), 2, tipLength=0.3)
    return states, frame

def get_red_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1) | cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


# ═══════════════════════════════════════════════════
#              GEOMETRY
# ═══════════════════════════════════════════════════

# Cache each goal post individually so one being occluded doesn't lose both
cached_goal_a = None  # last known position of GOAL_A
cached_goal_b = None  # last known position of GOAL_B

def update_goal_cache(s):
    global cached_goal_a, cached_goal_b
    if GOAL_A_ID in s:
        cached_goal_a = s[GOAL_A_ID]["center"]
    if GOAL_B_ID in s:
        cached_goal_b = s[GOAL_B_ID]["center"]

def get_goal_midpoint(s):
    """Get goal midpoint using live data + cached fallback for each post."""
    a = s[GOAL_A_ID]["center"] if GOAL_A_ID in s else cached_goal_a
    b = s[GOAL_B_ID]["center"] if GOAL_B_ID in s else cached_goal_b
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

def behind_trolley_position(s, gm):
    if gm is None or TROLLEY_ID not in s:
        return None
    tx, ty = s[TROLLEY_ID]["center"]
    dx, dy = tx - gm[0], ty - gm[1]
    d = math.sqrt(dx**2 + dy**2)
    if d < 1:
        return None
    ux, uy = dx / d, dy / d
    return (tx + ux * LINEUP_OFFSET_PX, ty + uy * LINEUP_OFFSET_PX)

def push_angle_from(s, gm):
    if gm is None or TROLLEY_ID not in s:
        return None
    tx, ty = s[TROLLEY_ID]["center"]
    dx = gm[0] - tx
    dy = -(gm[1] - ty)
    return math.degrees(math.atan2(dy, dx))

def dpx(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_behind(s, gm):
    b = behind_trolley_position(s, gm)
    if b is None or BOT_ID not in s:
        return False, 999
    bx, by = s[BOT_ID]["center"]
    dist_cells = dpx((bx, by), b) / CHUNK
    return dist_cells < BEHIND_TOLERANCE_CELLS, dist_cells

def is_facing_goal(s, gm):
    t = push_angle_from(s, gm)
    if t is None or BOT_ID not in s:
        return False, 999
    h = s[BOT_ID]["heading"]
    diff = abs((t - h + 180) % 360 - 180)
    return diff < ANGLE_TOLERANCE_DEG, diff

def trolley_at_goal(s, gm):
    if gm is None or TROLLEY_ID not in s:
        return False, 999
    tx, ty = s[TROLLEY_ID]["center"]
    dist_cells = dpx((tx, ty), gm) / CHUNK
    return dist_cells < PUSH_GOAL_REACHED_CELLS, dist_cells

def off_center_distance(s, gm):
    """
    How far the bot is from the ideal push line (trolley→goal extended behind).
    Returns perpendicular distance in cells, or 999 if can't calculate.
    """
    if gm is None or TROLLEY_ID not in s or BOT_ID not in s:
        return 999
    tx, ty = s[TROLLEY_ID]["center"]
    bx, by = s[BOT_ID]["center"]

    # Push direction unit vector (trolley → goal)
    dx = gm[0] - tx
    dy = gm[1] - ty
    d = math.sqrt(dx**2 + dy**2)
    if d < 1:
        return 999
    ux, uy = dx / d, dy / d

    # Vector from trolley to bot
    vx = bx - tx
    vy = by - ty

    # Perpendicular distance = |cross product|
    perp = abs(vx * uy - vy * ux)
    return perp / CHUNK


# ═══════════════════════════════════════════════════
#              PATHFINDING (threaded)
# ═══════════════════════════════════════════════════

path_lock = threading.Lock()
latest_path = []
path_computing = False

def build_inflated_grid(frame, states):
    marker_centers = [st["center"] for st in states.values()]
    red_mask = get_red_mask(frame)
    edges = cv2.Canny(red_mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = edges.shape
    CH, CW = H // CHUNK, W // CHUNK
    grid = np.zeros((CH, CW), dtype=int)
    mask = np.zeros_like(edges)

    for cnt in contours:
        cv2.drawContours(mask, [cnt], -1, 255, 1)

    for (mx, my) in marker_centers:
        cv2.circle(mask, (mx, my), 60, 0, -1)

    ys, xs = np.where(mask == 255)
    for x, y in zip(xs, ys):
        grid[min(y // CHUNK, CH - 1), min(x // CHUNK, CW - 1)] = 1

    inflate = BOT_INFLATE_CELLS if BOT_INFLATE_CELLS else 5
    gi = (grid.astype(np.uint8)) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflate*2+1, inflate*2+1))
    ig = (cv2.dilate(gi, k) > 0).astype(int)

    return ig, CH, CW

def compute_path_to(frame, s, target_px):
    if BOT_ID not in s:
        return []

    ig, CH, CW = build_inflated_grid(frame, s)

    bx, by = s[BOT_ID]["center"]
    start = (max(0, min(by // CHUNK, CH-1)), max(0, min(bx // CHUNK, CW-1)))
    goal  = (max(0, min(int(target_px[1]) // CHUNK, CH-1)),
             max(0, min(int(target_px[0]) // CHUNK, CW-1)))

    clear_cells(ig, start, 4)
    clear_cells(ig, goal, 5)

    if TROLLEY_ID in s:
        tx, ty = s[TROLLEY_ID]["center"]
        clear_cells(ig, (ty // CHUNK, tx // CHUNK), 6)

    raw = astar(ig, start, goal)
    if raw:
        return smooth_path(ig, raw)
    return []

def compute_path_async(frame, states, target_px):
    global path_computing
    if path_computing:
        return

    def worker():
        global latest_path, path_computing
        path_computing = True
        try:
            result = compute_path_to(frame, states, target_px)
            with path_lock:
                latest_path = result
        finally:
            path_computing = False

    t = threading.Thread(target=worker, daemon=True)
    t.start()

def get_latest_path():
    with path_lock:
        return list(latest_path)

def clear_cells(grid, pos, radius=2):
    rows, cols = grid.shape
    r, c = pos
    for rr in range(max(0, r - radius), min(rows, r + radius + 1)):
        for cc in range(max(0, c - radius), min(cols, c + radius + 1)):
            grid[rr, cc] = 0

def get_next_commands(path, heading_deg):
    if not path or len(path) < 2:
        return []
    r0, c0 = path[0]
    r1, c1 = path[1]
    dx = (c1 - c0) * CHUNK
    dy = (r0 - r1) * CHUNK
    target = math.degrees(math.atan2(dy, dx))
    turn = (target - heading_deg + 180) % 360 - 180
    dist = math.sqrt(dx**2 + dy**2) * MM_PER_PIXEL * DISTANCE_SCALE
    cmds = []
    if abs(turn) > 1:
        cmds.append(('ROTATE', round(turn, 1)))
    if dist > 1:
        cmds.append(('MOVE', round(dist, 1)))
    return cmds

def direct_move_to(s, target_px):
    if BOT_ID not in s:
        return []
    bx, by = s[BOT_ID]["center"]
    dx = target_px[0] - bx
    dy = -(target_px[1] - by)
    target_angle = math.degrees(math.atan2(dy, dx))
    heading = s[BOT_ID]["heading"]
    turn = (target_angle - heading + 180) % 360 - 180
    dist_mm = math.sqrt(dx**2 + dy**2) * MM_PER_PIXEL * DISTANCE_SCALE
    cmds = []
    if abs(turn) > 2:
        cmds.append(('ROTATE', round(turn, 1)))
    if dist_mm > 5:
        cmds.append(('MOVE', round(dist_mm, 1)))
    return cmds

def get_push_command(s, gm):
    t = push_angle_from(s, gm)
    if t is None or BOT_ID not in s:
        return []
    heading = s[BOT_ID]["heading"]
    turn = (t - heading + 180) % 360 - 180
    cmds = []
    if abs(turn) > 2:
        cmds.append(('ROTATE', round(turn, 1)))
    cmds.append(('MOVE', round(PUSH_STEP_MM, 1)))
    return cmds


# ═══════════════════════════════════════════════════
#              DISPLAY
# ═══════════════════════════════════════════════════

def draw_path_on_frame(disp, path):
    if not path or len(path) < 2:
        return
    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        px0 = int(c0 * CHUNK + CHUNK // 2)
        py0 = int(r0 * CHUNK + CHUNK // 2)
        px1 = int(c1 * CHUNK + CHUNK // 2)
        py1 = int(r1 * CHUNK + CHUNK // 2)
        cv2.line(disp, (px0, py0), (px1, py1), (255, 255, 0), 2, cv2.LINE_AA)
    for r, c in path:
        px = int(c * CHUNK + CHUNK // 2)
        py = int(r * CHUNK + CHUNK // 2)
        cv2.circle(disp, (px, py), 3, (0, 255, 255), -1)

def draw_overlay(disp, s, gm):
    behind = behind_trolley_position(s, gm)

    if TROLLEY_ID in s:
        tx, ty = s[TROLLEY_ID]["center"]
        r = int(TROLLEY_RADIUS_PX) if TROLLEY_RADIUS_PX else 30
        cv2.circle(disp, (int(tx), int(ty)), r, (0, 255, 255), 2)

    if BOT_ID in s and MM_PER_PIXEL:
        bx, by = s[BOT_ID]["center"]
        bot_r_px = int(BOT_RADIUS_MM / MM_PER_PIXEL)
        cv2.circle(disp, (int(bx), int(by)), bot_r_px, (255, 100, 0), 1)

    if gm:
        cv2.circle(disp, (int(gm[0]), int(gm[1])), 10, (0, 255, 0), -1)
        cv2.putText(disp, "GOAL", (int(gm[0]) + 15, int(gm[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if TROLLEY_ID in s:
            tx, ty = s[TROLLEY_ID]["center"]
            cv2.arrowedLine(disp, (int(tx), int(ty)), (int(gm[0]), int(gm[1])),
                            (0, 255, 0), 2, tipLength=0.1)

    if behind:
        cv2.circle(disp, (int(behind[0]), int(behind[1])), 8, (255, 0, 255), -1)
        cv2.putText(disp, "LINEUP", (int(behind[0]) + 12, int(behind[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        if BOT_ID in s:
            bx, by = s[BOT_ID]["center"]
            cv2.line(disp, (int(bx), int(by)), (int(behind[0]), int(behind[1])),
                     (255, 0, 255), 1, cv2.LINE_AA)

    # Draw goal post line using cached positions
    ga = s.get(GOAL_A_ID, {}).get("center", cached_goal_a)
    gb = s.get(GOAL_B_ID, {}).get("center", cached_goal_b)
    if ga and gb:
        cv2.line(disp, (int(ga[0]), int(ga[1])), (int(gb[0]), int(gb[1])), (0, 200, 0), 2)

    return disp


# ═══════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════

if __name__ == "__main__":

    robot_sock = None
    try:
        robot_sock = connect_robot(ROBOT_HOST, ROBOT_PORT)
    except Exception as e:
        print(f"Could not connect: {e}")
        print("Vision-only mode")

    cap = cv2.VideoCapture(1)

    with np.load('camera_params.npz') as data:
        mtx, dist_coeffs = data['mtx'], data['dist']
        dist_coeffs = dist_coeffs * 0.1

    first_frame = True
    newcameramtx, roi = None, None
    push_mode = False
    push_phase = False
    blind_push_count = 0

    # Stuck detection
    push_count_since_check = 0
    last_trolley_pos = None

    last_cmd_time = 0
    CMD_INTERVAL = 0.1

    print(f"Bot={BOT_ID} ({BOT_DIAMETER_MM}mm) Trolley={TROLLEY_ID} Goals={GOAL_A_ID},{GOAL_B_ID}")
    print(f"Tolerances: behind={BEHIND_TOLERANCE_CELLS}c angle={ANGLE_TOLERANCE_DEG}d "
          f"goal={PUSH_GOAL_REACHED_CELLS}c off-center={OFF_CENTER_THRESHOLD_CELLS}c")
    print(f"Stuck detect: {STUCK_PUSH_COUNT} pushes, {STUCK_MOVE_THRESHOLD_CELLS}c min move")
    print()
    print("C=start push  S=stop  R=red preview  Q=quit")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if first_frame:
            h, w = frame.shape[:2]
            print(f"Camera: {w}x{h}")
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist_coeffs, (w, h), 1, (w, h))
            first_frame = False

        undistorted = cv2.undistort(frame, mtx, dist_coeffs, None, newcameramtx)
        x, y, rw, rh = roi
        undistorted = undistorted[y:y+rh, x:x+rw]

        if MM_PER_PIXEL is None:
            ah, aw = undistorted.shape[:2]
            MM_PER_PIXEL = ARENA_WIDTH_MM / aw
            MM_PER_CELL = MM_PER_PIXEL * CHUNK
            TROLLEY_RADIUS_PX = TROLLEY_RADIUS_MM / MM_PER_PIXEL
            LINEUP_OFFSET_PX = LINEUP_OFFSET_MM / MM_PER_PIXEL
            BOT_INFLATE_CELLS = int(math.ceil(BOT_RADIUS_MM / MM_PER_CELL)) + 1
            print(f"Image: {aw}x{ah}  Scale: {MM_PER_PIXEL:.4f} mm/px  Cell: {MM_PER_CELL:.2f} mm")
            print(f"Bot inflate: {BOT_INFLATE_CELLS} cells  Lineup: {LINEUP_OFFSET_PX:.0f}px")
            print()

        # ─── Detect on full brightness ───
        live_states = detect_markers(undistorted)
        update_goal_cache(live_states)
        gm = get_goal_midpoint(live_states)

        has_bot = BOT_ID in live_states
        has_trolley = TROLLEY_ID in live_states
        has_goal = gm is not None

        # ─── Async path for display ───
        if has_bot and has_trolley and has_goal and not push_phase:
            behind = behind_trolley_position(live_states, gm)
            if behind:
                compute_path_async(undistorted.copy(), live_states, behind)

        # ─── Dimmed display ───
        live = (undistorted.astype(np.float32) * LIVE_DIM).astype(np.uint8)
        _, live = detect_and_draw_markers(live)
        live = draw_overlay(live, live_states, gm)
        draw_path_on_frame(live, get_latest_path())

        # ─── Status ───
        required = {BOT_ID, TROLLEY_ID, GOAL_A_ID, GOAL_B_ID}
        detected = required.intersection(live_states.keys())

        if has_bot and has_trolley and has_goal:
            bh_ok, bh_dist = is_behind(live_states, gm)
            fc_ok, fc_diff = is_facing_goal(live_states, gm)
            tg_ok, tg_dist = trolley_at_goal(live_states, gm)
            oc_dist = off_center_distance(live_states, gm)
            phase_str = "DONE" if tg_ok else ("PUSH" if push_phase else "LINEUP")
            status = (f"IDs:{len(detected)}/4 | {phase_str} | "
                      f"Behind:{bh_dist:.0f}c | Angle:{fc_diff:.0f}d | "
                      f"T->Goal:{tg_dist:.0f}c | OffC:{oc_dist:.0f}c")
            sc = (0, 255, 0)
        else:
            missing = required - detected
            status = f"IDs:{len(detected)}/4 | MISSING: {missing}"
            if has_goal:
                status += " (goal cached)"
            sc = (0, 0, 255)

        mc = (0, 165, 255) if push_mode else (180, 180, 180)
        if push_mode and push_phase:
            phase_label = "PUSHING"
        elif push_mode:
            phase_label = "LINING UP"
        else:
            phase_label = "IDLE"

        cv2.putText(live, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sc, 1)
        cv2.putText(live, phase_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mc, 2)
        if blind_push_count > 0:
            cv2.putText(live, f"Blind: {blind_push_count}/{MAX_BLIND_PUSHES}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

        cv2.imshow("Live", resize_to_fit(live))

        # ═══════════════════════════════════════════
        #              PUSH LOGIC
        # ═══════════════════════════════════════════
        now = time.time()

        if push_mode and robot_sock and (now - last_cmd_time) > CMD_INTERVAL:

            # ── Goal check ──
            if has_trolley and has_goal:
                tg_ok, tg_dist = trolley_at_goal(live_states, gm)
                if tg_ok:
                    print(f"\n=== TROLLEY AT GOAL ({tg_dist:.0f}c) ===\n")
                    try:
                        send_command(robot_sock, 'S', 0)
                    except:
                        pass
                    push_mode = False
                    push_phase = False
                    blind_push_count = 0
                    push_count_since_check = 0
                    last_trolley_pos = None
                    key = cv2.waitKey(1) & 0xFF
                    continue

            # ── PUSH PHASE ──
            if push_phase:
                if has_bot and has_trolley and has_goal:
                    blind_push_count = 0
                    bh_ok, bh_dist = is_behind(live_states, gm)
                    fc_ok, fc_diff = is_facing_goal(live_states, gm)
                    oc_dist = off_center_distance(live_states, gm)

                    # ── Stuck detection ──
                    tx, ty = live_states[TROLLEY_ID]["center"]
                    push_count_since_check += 1

                    if push_count_since_check >= STUCK_PUSH_COUNT:
                        if last_trolley_pos is not None:
                            moved = dpx((tx, ty), last_trolley_pos) / CHUNK
                            if moved < STUCK_MOVE_THRESHOLD_CELLS:
                                print(f"[STUCK] Trolley moved only {moved:.1f}c in "
                                      f"{STUCK_PUSH_COUNT} pushes — backing up to realign")
                                send_move(robot_sock, 'MOVE', -BACKUP_MM)  # won't work as negative
                                # Use backward command instead
                                send_command(robot_sock, 'B', BACKUP_MM)
                                last_cmd_time = time.time()
                                push_phase = False  # go back to lineup
                                push_count_since_check = 0
                                last_trolley_pos = (tx, ty)
                                continue
                        last_trolley_pos = (tx, ty)
                        push_count_since_check = 0

                    # ── Off-center check ──
                    if oc_dist > OFF_CENTER_THRESHOLD_CELLS:
                        print(f"[OFF-CENTER] {oc_dist:.0f}c off push line — backing up to realign")
                        send_command(robot_sock, 'B', BACKUP_MM)
                        last_cmd_time = time.time()
                        push_phase = False  # go back to lineup
                        continue

                    # ── Drifted too far ──
                    if bh_dist > BEHIND_TOLERANCE_CELLS * 2.5:
                        print(f"[PUSH] Drifted ({bh_dist:.0f}c), backing up then LINEUP")
                        send_command(robot_sock, 'B', BACKUP_MM)
                        last_cmd_time = time.time()
                        push_phase = False
                        continue

                    # ── Normal push ──
                    cmds = get_push_command(live_states, gm)
                    print(f"[PUSH] behind={bh_dist:.0f}c angle={fc_diff:.0f}d "
                          f"offcenter={oc_dist:.0f}c -> {cmds}")
                    for ct, v in cmds:
                        if not send_move(robot_sock, ct, v):
                            push_mode = False
                            push_phase = False
                            break
                    last_cmd_time = time.time()

                elif has_goal:
                    blind_push_count += 1
                    if blind_push_count <= MAX_BLIND_PUSHES:
                        print(f"[BLIND {blind_push_count}/{MAX_BLIND_PUSHES}] Forward {PUSH_STEP_MM}mm")
                        send_move(robot_sock, 'MOVE', PUSH_STEP_MM)
                        last_cmd_time = time.time()
                    else:
                        print("[BLIND] Max reached — backing up to find markers")
                        send_command(robot_sock, 'B', BACKUP_MM)
                        last_cmd_time = time.time()
                        push_phase = False
                        blind_push_count = 0

            # ── LINEUP PHASE ──
            elif not push_phase:
                if has_bot and has_trolley and has_goal:
                    blind_push_count = 0
                    bh_ok, bh_dist = is_behind(live_states, gm)

                    if bh_ok:
                        print(f"[LINEUP] Behind! ({bh_dist:.0f}c) -> PUSH")
                        push_phase = True
                        push_count_since_check = 0
                        if has_trolley:
                            tx, ty = live_states[TROLLEY_ID]["center"]
                            last_trolley_pos = (tx, ty)
                    else:
                        behind = behind_trolley_position(live_states, gm)
                        if behind:
                            path = get_latest_path()
                            if not path or len(path) < 2:
                                path = compute_path_to(undistorted.copy(), live_states, behind)

                            if path and len(path) >= 2:
                                cmds = get_next_commands(path, live_states[BOT_ID]["heading"])
                                print(f"[LINEUP] {bh_dist:.0f}c | {len(path)} pts -> {cmds}")
                                for ct, v in cmds:
                                    if not send_move(robot_sock, ct, v):
                                        push_mode = False
                                        break
                                last_cmd_time = time.time()
                            else:
                                cmds = direct_move_to(live_states, behind)
                                print(f"[LINEUP] Direct -> {cmds}")
                                for ct, v in cmds:
                                    if not send_move(robot_sock, ct, v):
                                        push_mode = False
                                        break
                                last_cmd_time = time.time()

                elif not has_bot:
                    print(f"[WAIT] Bot {BOT_ID} not detected")
                elif not has_trolley:
                    print(f"[WAIT] Trolley {TROLLEY_ID} not detected")

        # ─── Keys ───
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            push_mode = False
            push_phase = False
            blind_push_count = 0
            push_count_since_check = 0
            last_trolley_pos = None
            if robot_sock:
                try:
                    send_command(robot_sock, 'S', 0)
                except:
                    pass
            print("STOP")
        if key == ord('c'):
            if robot_sock:
                push_mode = not push_mode
                if push_mode:
                    push_phase = False
                    blind_push_count = 0
                    push_count_since_check = 0
                    last_trolley_pos = None
                    print(f"\n=== PUSH MODE ON ===\n")
                else:
                    push_phase = False
                    blind_push_count = 0
                    push_count_since_check = 0
                    last_trolley_pos = None
                    print(f"\n=== PUSH MODE OFF ===\n")
                    try:
                        send_command(robot_sock, 'S', 0)
                    except:
                        pass
            else:
                print("No robot")
        if key == ord('r'):
            rm = get_red_mask(undistorted.copy())
            cv2.imshow("Red", resize_to_fit(np.hstack([
                resize_to_fit(undistorted, 640, 360),
                resize_to_fit(cv2.cvtColor(rm, cv2.COLOR_GRAY2BGR), 640, 360)
            ])))

    cap.release()
    cv2.destroyAllWindows()
    if robot_sock:
        try:
            send_command(robot_sock, 'S', 0)
        except:
            pass
        robot_sock.close()
