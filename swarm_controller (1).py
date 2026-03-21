import cv2 as cv
import numpy as np
import socket
import heapq
import time
from enum import Enum
from collections import deque

# ─────────────────────────────────────────────────────────────
# CONFIG — update these on the day
# ─────────────────────────────────────────────────────────────

MONA_IPS = {
    0: "192.168.1.101",
    1: "192.168.1.102",
    2: "192.168.1.103",
}
UDP_PORT     = 5005
CAMERA_INDEX = 0

ARENA_W = 1280
ARENA_H = 720

# Real-world sizes in CM
BOT_R_CM     = 4.0
TROLLEY_R_CM = 4.5    # update once you measure the trolley on the day
OBS_R_CM     = 2.5

PIXELS_PER_CM = 8.0   # measure on the day: put ruler in frame, count pixels

# Derived pixel sizes
BOT_R_PX     = int(BOT_R_CM     * PIXELS_PER_CM)
TROLLEY_R_PX = int(TROLLEY_R_CM * PIXELS_PER_CM)
OBS_R_PX     = int(OBS_R_CM     * PIXELS_PER_CM)

# Contact distance: centre-to-centre when bot just touches trolley
# = bot radius + trolley radius + 0.5cm gap so they just touch without overlap
CONTACT_PX = int((BOT_R_CM + TROLLEY_R_CM + 0.5) * PIXELS_PER_CM)

# Side bot offset — bots sit 90 degrees to the sides of the trolley
# Separation between the two side bots = 2 * SIDE_Y_PX
# Must be > 2 * BOT_R_PX (8cm) so side bots don't collide with each other
SIDE_Y_PX = CONTACT_PX   # ~9cm > 8cm bot diameter — just clear

FINISH_LINE_X = ARENA_W - 80
START_LINE_X  = 80

BOT_IDS    = [0, 1, 2]
TROLLEY_ID = 9
DICT_TYPE  = cv.aruco.DICT_4X4_50

# Control
BASE_SPEED     = 80
TURN_SPEED     = 50    # slower speed used during SWAP manoeuvre
TURN_GAIN      = 1.5
AT_TARGET_PX   = int(3.0 * PIXELS_PER_CM)
OBS_MARGIN_PX  = int(8   * PIXELS_PER_CM)
EDGE_MARGIN_PX = BOT_R_PX + 10
DRIFT_THRESH   = 20    # degrees of trolley drift before correction

# Turnaround: if A* path requires the trolley to travel more than this
# angle off the push axis we trigger a SWAP to flip the rear bot
SWAP_ANGLE_THRESH = 60  # degrees

# A* grid
ASTAR_CELL_CM   = 3
ASTAR_CELL_PX   = int(ASTAR_CELL_CM * PIXELS_PER_CM)
REPLAN_INTERVAL = 0.4   # seconds between replans

# Red obstacle HSV — tune with T key before first run
RED_LOWER_1 = np.array([0, 171, 98])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 171, 98])
RED_UPPER_2 = np.array([180, 255, 255])
MIN_OBS_AREA = 200

# ─────────────────────────────────────────────────────────────
# UDP COMMS
# ─────────────────────────────────────────────────────────────

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_command(bot_id: int, left: int, right: int):
    if bot_id not in MONA_IPS:
        return
    left  = int(np.clip(left,  -100, 100))
    right = int(np.clip(right, -100, 100))
    sock.sendto(f"{bot_id},{left},{right}".encode(), (MONA_IPS[bot_id], UDP_PORT))

def stop_all():
    for bot_id in MONA_IPS:
        send_command(bot_id, 0, 0)

# ─────────────────────────────────────────────────────────────
# VISION — ArUco detection
# ─────────────────────────────────────────────────────────────

aruco_dict   = cv.aruco.getPredefinedDictionary(DICT_TYPE)
aruco_params = cv.aruco.DetectorParameters()
detector     = cv.aruco.ArucoDetector(aruco_dict, aruco_params)

def get_center(corners_single):
    pts = corners_single[0]
    return int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))

def get_heading(corners_single):
    pts     = corners_single[0]
    top_mid = (pts[0] + pts[1]) / 2.0
    center  = np.mean(pts, axis=0)
    delta   = top_mid - center
    return float(np.degrees(np.arctan2(-delta[1], delta[0])))

def detect_markers(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    bot_states  = {}
    trolley_pos = None
    trolley_hdg = None
    if ids is None:
        return bot_states, trolley_pos, trolley_hdg
    cv.aruco.drawDetectedMarkers(frame, corners, ids)
    for i, mid in enumerate(ids.flatten()):
        cx_p, cy_p = get_center(corners[i])
        hdg        = get_heading(corners[i])
        if mid in BOT_IDS:
            bot_states[mid] = {"center": (cx_p, cy_p), "heading": hdg}
        elif mid == TROLLEY_ID:
            trolley_pos = (cx_p, cy_p)
            trolley_hdg = hdg
    return bot_states, trolley_pos, trolley_hdg

# ─────────────────────────────────────────────────────────────
# VISION — Red obstacle detection
# ─────────────────────────────────────────────────────────────

def detect_obstacles(frame):
    hsv  = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = (cv.inRange(hsv, RED_LOWER_1, RED_UPPER_1) |
            cv.inRange(hsv, RED_LOWER_2, RED_UPPER_2))
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv.morphologyEx(mask, cv.MORPH_OPEN,  kernel)
    mask   = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    obstacles = []
    for cnt in contours:
        if cv.contourArea(cnt) < MIN_OBS_AREA:
            continue
        (ox, oy), radius = cv.minEnclosingCircle(cnt)
        radius = int(radius) + OBS_MARGIN_PX // 2
        obstacles.append((int(ox), int(oy), radius))
        cv.circle(frame, (int(ox), int(oy)), radius, (0, 0, 255), 2)
    return obstacles

def near_edge(pos):
    x, y = pos
    return (x < EDGE_MARGIN_PX or x > ARENA_W - EDGE_MARGIN_PX or
            y < EDGE_MARGIN_PX or y > ARENA_H - EDGE_MARGIN_PX)

# ─────────────────────────────────────────────────────────────
# A* PATHFINDING
# ─────────────────────────────────────────────────────────────

GRID_COLS = ARENA_W // ASTAR_CELL_PX
GRID_ROWS = ARENA_H // ASTAR_CELL_PX

def px_to_cell(px, py):
    return (max(0, min(GRID_COLS-1, int(px // ASTAR_CELL_PX))),
            max(0, min(GRID_ROWS-1, int(py // ASTAR_CELL_PX))))

def cell_to_px(col, row):
    return (int((col + 0.5) * ASTAR_CELL_PX),
            int((row + 0.5) * ASTAR_CELL_PX))

def build_grid(obstacles, inflation_px, extra_edge_px=0):
    """
    Bool grid: True = passable.
    inflation_px: obstacle radius inflation = agent radius (config space).
    extra_edge_px: additional wall keep-out (used during push so trolley
                   body stays off arena walls).
    """
    grid      = np.ones((GRID_ROWS, GRID_COLS), dtype=bool)
    edge_keep = EDGE_MARGIN_PX + extra_edge_px
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            cx_c, cy_c = cell_to_px(col, row)
            if (cx_c < edge_keep or cx_c > ARENA_W - edge_keep or
                    cy_c < edge_keep or cy_c > ARENA_H - edge_keep):
                grid[row, col] = False
                continue
            for (ox, oy, obs_r) in obstacles:
                if np.hypot(cx_c - ox, cy_c - oy) < obs_r + inflation_px:
                    grid[row, col] = False
                    break
    return grid

def nearest_free_cell(grid, cell):
    q       = deque([cell])
    visited = {cell}
    while q:
        c, r = q.popleft()
        if 0 <= c < GRID_COLS and 0 <= r < GRID_ROWS and grid[r, c]:
            return (c, r)
        for dc, dr in [(1,0),(-1,0),(0,1),(0,-1)]:
            nc, nr = c+dc, r+dr
            if (0 <= nc < GRID_COLS and 0 <= nr < GRID_ROWS
                    and (nc,nr) not in visited):
                visited.add((nc,nr)); q.append((nc,nr))
    return None

def line_of_sight(grid, p1, p2):
    c1, r1 = px_to_cell(*p1);  c2, r2 = px_to_cell(*p2)
    dc = abs(c2-c1); dr = abs(r2-r1)
    sc = 1 if c1 < c2 else -1; sr = 1 if r1 < r2 else -1
    err = dc - dr;  c, r = c1, r1
    while True:
        if not (0 <= c < GRID_COLS and 0 <= r < GRID_ROWS): return False
        if not grid[r, c]: return False
        if c == c2 and r == r2: return True
        e2 = 2*err
        if e2 > -dr: err -= dr; c += sc
        if e2 <  dc: err += dc; r += sr

def smooth_path(path, grid):
    if len(path) <= 2:
        return path
    smoothed = [path[0]]; i = 0
    while i < len(path)-1:
        j = len(path)-1
        while j > i+1:
            if line_of_sight(grid, path[i], path[j]): break
            j -= 1
        smoothed.append(path[j]); i = j
    return smoothed

def astar(grid, start_px, goal_px):
    """
    A* on inflated grid. Returns smoothed waypoint list or [].
    """
    sc = px_to_cell(*start_px); gc = px_to_cell(*goal_px)
    if not grid[sc[1], sc[0]]: sc = nearest_free_cell(grid, sc)
    if not grid[gc[1], gc[0]]: gc = nearest_free_cell(grid, gc)
    if sc is None or gc is None: return []
    if sc == gc: return [cell_to_px(*gc)]

    def h(a, b): return np.hypot(a[0]-b[0], a[1]-b[1]) * ASTAR_CELL_PX

    heap = [(0.0, sc)]; came_from = {}; g = {sc: 0.0}
    MOVES = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),
             (1,1,1.414),(-1,1,1.414),(1,-1,1.414),(-1,-1,1.414)]

    while heap:
        _, cur = heapq.heappop(heap)
        if cur == gc:
            path = []; node = cur
            while node in came_from:
                path.append(cell_to_px(*node)); node = came_from[node]
            path.append(cell_to_px(*sc)); path.reverse()
            return smooth_path(path, grid)
        for dc, dr, cost in MOVES:
            nb = (cur[0]+dc, cur[1]+dr)
            if not (0 <= nb[0] < GRID_COLS and 0 <= nb[1] < GRID_ROWS): continue
            if not grid[nb[1], nb[0]]: continue
            ng = g[cur] + cost * ASTAR_CELL_PX
            if ng < g.get(nb, float('inf')):
                came_from[nb] = cur; g[nb] = ng
                heapq.heappush(heap, (ng + h(nb, gc), nb))
    return []

# ─────────────────────────────────────────────────────────────
# PATH CACHE
# ─────────────────────────────────────────────────────────────

class PathCache:
    def __init__(self):
        self.paths = {}; self.last_plan = {}; self.waypoint = {}

    def needs_replan(self, bot_id):
        return time.time() - self.last_plan.get(bot_id, 0) > REPLAN_INTERVAL

    def set_path(self, bot_id, path):
        self.paths[bot_id] = path
        self.last_plan[bot_id] = time.time()
        self.waypoint[bot_id] = 0

    def advance(self, bot_id, pos):
        path = self.paths.get(bot_id, [])
        idx  = self.waypoint.get(bot_id, 0)
        if path and idx < len(path):
            if np.hypot(pos[0]-path[idx][0], pos[1]-path[idx][1]) < AT_TARGET_PX:
                self.waypoint[bot_id] = min(idx+1, len(path)-1)

    def current_wp(self, bot_id):
        path = self.paths.get(bot_id, [])
        idx  = self.waypoint.get(bot_id, 0)
        return path[idx] if path and idx < len(path) else None

    def clear(self, bot_id=None):
        if bot_id is None:
            self.paths.clear(); self.last_plan.clear(); self.waypoint.clear()
        else:
            self.paths.pop(bot_id,None); self.last_plan.pop(bot_id,None)
            self.waypoint.pop(bot_id,None)

path_cache = PathCache()

# ─────────────────────────────────────────────────────────────
# FORMATION
# ─────────────────────────────────────────────────────────────
#
# Normal T-formation — push direction is rightward (+x):
#
#          [B1]   ← side bot, upper
#            |
#   [B0] ── [TROLLEY] ──────► goal
#            |
#          [B2]   ← side bot, lower
#
# B0 pushes from directly behind (rear bot).
# B1 sits at 90° to the side (top).
# B2 sits at 90° to the other side (bottom).
#
# The side bots stabilise lateral drift without fighting each other.
# Turning is easier because differential speed of B1 vs B2 steers the
# trolley left/right — like differential drive.
#
# Bot separation check:
#   B0 ↔ B1 = hypot(CONTACT_PX, SIDE_Y_PX) > 2*BOT_R_PX ✓
#   B1 ↔ B2 = 2 * SIDE_Y_PX > 2*BOT_R_PX ✓
#
# push_angle_deg: the direction the formation should push, in degrees.
# 0 = right, 90 = up, 180 = left, -90 = down.
# Normally 0 (straight to finish), but changed during SWAP to steer around.

def get_push_formation(trolley_pos, push_angle_deg=0.0):
    """
    Returns {bot_id: (target_px, target_py)} for the T-formation.
    push_angle_deg: direction of travel in image-space degrees
                    (0=right, 90=up in screen coords).
    The rear bot (B0) is always opposite to push_angle.
    The side bots (B1, B2) are 90 degrees to either side.
    """
    tx, ty = trolley_pos
    rad    = np.radians(push_angle_deg)

    # Unit vectors
    fwd  = np.array([ np.cos(rad), -np.sin(rad)])  # forward (push direction)
    left = np.array([-np.sin(rad), -np.cos(rad)])  # 90° left of forward

    # B0: directly behind (opposite of push direction)
    b0 = (int(tx - fwd[0]*CONTACT_PX),
          int(ty - fwd[1]*CONTACT_PX))

    # B1: to the left of trolley centre, perpendicular to push axis
    b1 = (int(tx + left[0]*SIDE_Y_PX),
          int(ty + left[1]*SIDE_Y_PX))

    # B2: to the right of trolley centre (opposite side)
    b2 = (int(tx - left[0]*SIDE_Y_PX),
          int(ty - left[1]*SIDE_Y_PX))

    return {0: b0, 1: b1, 2: b2}

def all_in_formation(bot_states, targets):
    return all(
        bot_id in bot_states and
        np.hypot(bot_states[bot_id]["center"][0] - tgt[0],
                 bot_states[bot_id]["center"][1] - tgt[1]) < AT_TARGET_PX
        for bot_id, tgt in targets.items()
    )

# ─────────────────────────────────────────────────────────────
# TURNAROUND / SWAP LOGIC
# ─────────────────────────────────────────────────────────────
#
# When the A* path requires the trolley to make a large turn
# (path angle > SWAP_ANGLE_THRESH from current push axis), the rear
# bot (B0) needs to move to the opposite side of the trolley so it
# becomes a pusher in the new direction.
#
# Swap sequence:
#   1. SWAP_PULLBACK  — B0 backs away from trolley along current rear axis
#   2. SWAP_REPOSITION— B0 navigates around the trolley to the new rear slot
#   3. SWAP_REFORM    — all bots move to the new formation positions
#   4. Resume PUSH with updated push_angle
#
# During steps 1-3 the two side bots hold position to keep the trolley stable.

def angle_diff(a, b):
    """Signed difference between two angles in degrees, result in -180..180."""
    return ((a - b) + 180) % 360 - 180

def compute_push_angle(trolley_pos, goal_pos):
    """
    Desired push angle from trolley toward goal, in image-space degrees.
    0 = rightward, 90 = upward, etc.
    """
    dx = goal_pos[0] - trolley_pos[0]
    dy = goal_pos[1] - trolley_pos[1]
    return float(np.degrees(np.arctan2(-dy, dx)))

def needs_swap(current_push_angle, desired_push_angle):
    """
    Returns True if the push direction needs to change by more than
    SWAP_ANGLE_THRESH degrees — meaning the rear bot must reposition.
    """
    return abs(angle_diff(desired_push_angle, current_push_angle)) > SWAP_ANGLE_THRESH

# ─────────────────────────────────────────────────────────────
# CONTROL — steering
# ─────────────────────────────────────────────────────────────

def steer_to_waypoint(bot_id, pos, heading, wp, speed=BASE_SPEED):
    dx      = wp[0] - pos[0]
    dy      = wp[1] - pos[1]
    desired = np.degrees(np.arctan2(-dy, dx))
    error   = (desired - heading + 180) % 360 - 180
    corr    = int(error * TURN_GAIN)
    send_command(bot_id,
                 int(np.clip(speed - corr, -100, 100)),
                 int(np.clip(speed + corr, -100, 100)))

def navigate_bot(bot_id, pos, heading, goal, obstacles,
                 inflation_px, speed=BASE_SPEED, extra_edge=0):
    """
    Replan A* if needed, then steer toward current waypoint.
    A* guarantees a collision-free path so no repulsion is needed here.
    """
    path_cache.advance(bot_id, pos)

    if path_cache.needs_replan(bot_id):
        grid = build_grid(obstacles, inflation_px, extra_edge_px=extra_edge)
        path = astar(grid, pos, goal)
        if path:
            path_cache.set_path(bot_id, path)
        else:
            send_command(bot_id, -40, -40)
            return

    wp = path_cache.current_wp(bot_id)
    if wp is None:
        send_command(bot_id, 0, 0)
        return

    dist_goal = np.hypot(pos[0]-goal[0], pos[1]-goal[1])
    eff_speed = max(40, int(speed * min(1.0, dist_goal / (AT_TARGET_PX * 3))))
    steer_to_waypoint(bot_id, pos, heading, wp, eff_speed)

# ─────────────────────────────────────────────────────────────
# STATE MACHINE
# ─────────────────────────────────────────────────────────────

class State(Enum):
    TRAVEL         = 0   # bots navigate from finish to T-formation around trolley
    FORM           = 1   # fine-adjust into exact formation positions
    PUSH           = 2   # T-formation pushes trolley toward finish
    SWAP_PULLBACK  = 3   # B0 backs away from trolley before repositioning
    SWAP_REPOSITION= 4   # B0 navigates to new rear slot around trolley
    SWAP_REFORM    = 5   # all bots settle into new formation
    DONE           = 6

state            = State.TRAVEL
targets          = {}
push_angle       = 0.0   # current push direction in degrees (0 = rightward)
swap_new_angle   = 0.0   # target push angle after swap
swap_pullback_tgt= None  # where B0 retreats to before repositioning

def update_state(bot_states, trolley_pos, trolley_hdg, obstacles):
    global state, targets, push_angle, swap_new_angle, swap_pullback_tgt

    # ── TRAVEL ──────────────────────────────────────────────────
    # Each bot uses A* independently to reach its formation slot.
    # Inflation = BOT_R_PX (single bot navigating alone).
    if state == State.TRAVEL:
        if trolley_pos is None:
            for bot_id, s in bot_states.items():
                navigate_bot(bot_id, s["center"], s["heading"],
                             (START_LINE_X, ARENA_H // 2),
                             obstacles, BOT_R_PX)
            return

        targets     = get_push_formation(trolley_pos, push_angle)
        all_arrived = True

        for bot_id, tgt in targets.items():
            if bot_id not in bot_states:
                all_arrived = False; continue
            pos = bot_states[bot_id]["center"]
            hdg = bot_states[bot_id]["heading"]
            if near_edge(pos):
                send_command(bot_id, -50, -50)
                all_arrived = False; continue
            if np.hypot(pos[0]-tgt[0], pos[1]-tgt[1]) > AT_TARGET_PX:
                navigate_bot(bot_id, pos, hdg, tgt, obstacles, BOT_R_PX)
                all_arrived = False
            else:
                send_command(bot_id, 0, 0)

        if all_arrived:
            print("[STATE] Formation reached → FORM")
            path_cache.clear(); state = State.FORM

    # ── FORM ────────────────────────────────────────────────────
    # Fine-adjust: recompute targets from live trolley pos and creep in.
    elif state == State.FORM:
        if trolley_pos is None:
            print("[STATE] Lost trolley → TRAVEL")
            path_cache.clear(); state = State.TRAVEL; return

        targets = get_push_formation(trolley_pos, push_angle)
        for bot_id, tgt in targets.items():
            if bot_id not in bot_states: continue
            navigate_bot(bot_id, bot_states[bot_id]["center"],
                         bot_states[bot_id]["heading"],
                         tgt, obstacles, BOT_R_PX, speed=50)

        if all_in_formation(bot_states, targets):
            print(f"[STATE] Formation locked (push angle={push_angle:.0f}°) → PUSH")
            path_cache.clear(); state = State.PUSH

    # ── PUSH ────────────────────────────────────────────────────
    # Push the trolley toward FINISH_LINE_X.
    # Formation inflation = BOT_R_PX + TROLLEY_R_PX so the path routes
    # the trolley body safely past obstacles, not just bot centres.
    # extra_edge = TROLLEY_R_PX keeps the trolley off arena walls.
    elif state == State.PUSH:
        if trolley_pos is None:
            print("[STATE] Lost trolley → FORM")
            path_cache.clear(); state = State.FORM; return

        if trolley_pos[0] >= FINISH_LINE_X:
            print("[STATE] DONE — trolley at finish!")
            stop_all(); state = State.DONE; return

        # Check if the direct path to finish now requires a large turn
        desired_angle = compute_push_angle(trolley_pos, (FINISH_LINE_X, ARENA_H//2))
        if needs_swap(push_angle, desired_angle):
            print(f"[STATE] Turn too large ({desired_angle:.0f}° vs {push_angle:.0f}°) → SWAP_PULLBACK")
            swap_new_angle = desired_angle
            # B0 pullback target: retreat CONTACT_PX*2 behind current rear slot
            rad = np.radians(push_angle)
            fwd = np.array([np.cos(rad), -np.sin(rad)])
            if trolley_pos:
                tx, ty = trolley_pos
                swap_pullback_tgt = (
                    int(tx - fwd[0] * CONTACT_PX * 2.5),
                    int(ty - fwd[1] * CONTACT_PX * 2.5)
                )
            path_cache.clear(); state = State.SWAP_PULLBACK; return

        targets = get_push_formation(trolley_pos, push_angle)

        # Drift correction: slow the side bot on the high side to steer back
        slow_id = None
        if trolley_hdg is not None and abs(angle_diff(trolley_hdg, push_angle)) > DRIFT_THRESH:
            drift = angle_diff(trolley_hdg, push_angle)
            slow_id = 1 if drift > 0 else 2   # B1=upper, B2=lower

        f_infl   = BOT_R_PX + TROLLEY_R_PX
        x_offset = FINISH_LINE_X - trolley_pos[0]

        for bot_id, s in bot_states.items():
            if bot_id not in targets: continue
            tgt  = targets[bot_id]
            goal = (tgt[0] + x_offset, tgt[1])
            spd  = BASE_SPEED // 2 if bot_id == slow_id else BASE_SPEED
            navigate_bot(bot_id, s["center"], s["heading"], goal,
                         obstacles, f_infl, speed=spd,
                         extra_edge=TROLLEY_R_PX)

        for bot_id, s in bot_states.items():
            if near_edge(s["center"]):
                send_command(bot_id, -50, -50)

    # ── SWAP_PULLBACK ────────────────────────────────────────────
    # B0 (rear bot) reverses away from the trolley.
    # B1 and B2 hold their side positions to keep trolley stable.
    elif state == State.SWAP_PULLBACK:
        if trolley_pos is None:
            state = State.TRAVEL; return

        # Hold side bots in place
        side_targets = get_push_formation(trolley_pos, push_angle)
        for bot_id in [1, 2]:
            if bot_id not in bot_states: continue
            tgt = side_targets[bot_id]
            navigate_bot(bot_id, bot_states[bot_id]["center"],
                         bot_states[bot_id]["heading"],
                         tgt, obstacles, BOT_R_PX, speed=40)

        # B0 backs away
        if 0 in bot_states and swap_pullback_tgt is not None:
            pos = bot_states[0]["center"]
            navigate_bot(0, pos, bot_states[0]["heading"],
                         swap_pullback_tgt, obstacles, BOT_R_PX, speed=TURN_SPEED)

            if np.hypot(pos[0]-swap_pullback_tgt[0],
                        pos[1]-swap_pullback_tgt[1]) < AT_TARGET_PX * 1.5:
                print("[STATE] B0 pulled back → SWAP_REPOSITION")
                path_cache.clear(0)
                state = State.SWAP_REPOSITION

    # ── SWAP_REPOSITION ──────────────────────────────────────────
    # B0 navigates around the trolley to the new rear slot.
    # Side bots continue holding their positions.
    elif state == State.SWAP_REPOSITION:
        if trolley_pos is None:
            state = State.TRAVEL; return

        # New rear slot based on the new push angle
        new_targets = get_push_formation(trolley_pos, swap_new_angle)
        new_b0_pos  = new_targets[0]

        # Hold side bots
        side_targets = get_push_formation(trolley_pos, push_angle)
        for bot_id in [1, 2]:
            if bot_id not in bot_states: continue
            navigate_bot(bot_id, bot_states[bot_id]["center"],
                         bot_states[bot_id]["heading"],
                         side_targets[bot_id], obstacles, BOT_R_PX, speed=40)

        # B0 navigates around trolley to new rear position
        # Inflation = BOT_R_PX + TROLLEY_R_PX so B0 arcs around, not through
        if 0 in bot_states:
            pos = bot_states[0]["center"]
            navigate_bot(0, pos, bot_states[0]["heading"],
                         new_b0_pos, obstacles,
                         BOT_R_PX + TROLLEY_R_PX,
                         speed=TURN_SPEED)

            if np.hypot(pos[0]-new_b0_pos[0],
                        pos[1]-new_b0_pos[1]) < AT_TARGET_PX * 1.5:
                print(f"[STATE] B0 at new rear slot → SWAP_REFORM  new angle={swap_new_angle:.0f}°")
                push_angle = swap_new_angle
                path_cache.clear()
                state = State.SWAP_REFORM

    # ── SWAP_REFORM ──────────────────────────────────────────────
    # All three bots settle into the new T-formation simultaneously.
    elif state == State.SWAP_REFORM:
        if trolley_pos is None:
            state = State.TRAVEL; return

        targets = get_push_formation(trolley_pos, push_angle)
        for bot_id, tgt in targets.items():
            if bot_id not in bot_states: continue
            navigate_bot(bot_id, bot_states[bot_id]["center"],
                         bot_states[bot_id]["heading"],
                         tgt, obstacles, BOT_R_PX, speed=50)

        if all_in_formation(bot_states, targets):
            print("[STATE] Swap complete → PUSH")
            path_cache.clear(); state = State.PUSH

    # ── DONE ────────────────────────────────────────────────────
    elif state == State.DONE:
        stop_all()

# ─────────────────────────────────────────────────────────────
# DEBUG OVERLAY
# ─────────────────────────────────────────────────────────────

PATH_COLOURS = {0: (255, 200, 0), 1: (200, 255, 0), 2: (0, 255, 200)}

def draw_debug(frame, bot_states, trolley_pos, trolley_hdg, obstacles):
    cv.rectangle(frame,
                 (EDGE_MARGIN_PX, EDGE_MARGIN_PX),
                 (ARENA_W - EDGE_MARGIN_PX, ARENA_H - EDGE_MARGIN_PX),
                 (200, 150, 0), 1)

    cv.line(frame, (START_LINE_X,  0), (START_LINE_X,  ARENA_H), (0, 255, 0),   2)
    cv.line(frame, (FINISH_LINE_X, 0), (FINISH_LINE_X, ARENA_H), (0, 100, 255), 2)
    cv.putText(frame, "START",  (START_LINE_X  + 5, 25),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),   2)
    cv.putText(frame, "FINISH", (FINISH_LINE_X + 5, 25),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

    if trolley_pos:
        cv.circle(frame, trolley_pos, TROLLEY_R_PX, (0, 255, 0), 2)
        lbl = f"TROLLEY {trolley_hdg:.0f}d" if trolley_hdg is not None else "TROLLEY"
        cv.putText(frame, lbl,
                   (trolley_pos[0]+TROLLEY_R_PX+4, trolley_pos[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw formation targets
        ftgts  = get_push_formation(trolley_pos, push_angle)
        labels = {0: "REAR", 1: "SIDE-L", 2: "SIDE-R"}
        for bot_id, tgt in ftgts.items():
            cv.circle(frame, tgt, BOT_R_PX, (0, 200, 255), 1)
            cv.putText(frame, labels[bot_id],
                       (tgt[0]+BOT_R_PX+2, tgt[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.32, (0, 200, 255), 1)

        # Draw push direction arrow from trolley centre
        rad = np.radians(push_angle)
        ax  = int(trolley_pos[0] + np.cos(rad) * TROLLEY_R_PX * 2.5)
        ay  = int(trolley_pos[1] - np.sin(rad) * TROLLEY_R_PX * 2.5)
        cv.arrowedLine(frame, trolley_pos, (ax, ay), (0, 255, 150), 2, tipLength=0.3)

    # A* paths
    for bot_id in BOT_IDS:
        path = path_cache.paths.get(bot_id, [])
        idx  = path_cache.waypoint.get(bot_id, 0)
        col  = PATH_COLOURS.get(bot_id, (200, 200, 200))
        for k in range(len(path)-1):
            cv.line(frame, path[k], path[k+1], col, 2 if k >= idx else 1)
        if path and idx < len(path):
            cv.circle(frame, path[idx], 6, col, -1)

    # Swap pullback target
    if swap_pullback_tgt and state in (State.SWAP_PULLBACK, State.SWAP_REPOSITION):
        cv.circle(frame, swap_pullback_tgt, 8, (255, 80, 200), -1)
        cv.putText(frame, "B0 retreat", (swap_pullback_tgt[0]+8, swap_pullback_tgt[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 80, 200), 1)

    # Bot arrows + labels
    for bot_id, s in bot_states.items():
        cx_p, cy_p = s["center"]
        hdg        = s["heading"]
        ex = int(cx_p + (BOT_R_PX+15)*np.cos(np.radians(hdg)))
        ey = int(cy_p - (BOT_R_PX+15)*np.sin(np.radians(hdg)))
        cv.arrowedLine(frame, (cx_p, cy_p), (ex, ey), (255, 100, 0), 2)
        cv.putText(frame, f"B{bot_id}",
                   (cx_p-10, cy_p+BOT_R_PX+14),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # State, push angle, grid info
    cv.putText(frame, f"State: {state.name}", (10, 35),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv.putText(frame, f"Push angle: {push_angle:.0f}deg",
               (10, 62), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv.putText(frame, f"Grid {GRID_COLS}x{GRID_ROWS}  cell={ASTAR_CELL_CM}cm",
               (10, 82), cv.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

# ─────────────────────────────────────────────────────────────
# HSV TUNER
# ─────────────────────────────────────────────────────────────

def run_hsv_tuner(cap):
    cv.namedWindow("HSV tuner")
    cv.createTrackbar("H lo1", "HSV tuner", 0,   10,  lambda x: None)
    cv.createTrackbar("H hi1", "HSV tuner", 10,  10,  lambda x: None)
    cv.createTrackbar("H lo2", "HSV tuner", 170, 180, lambda x: None)
    cv.createTrackbar("H hi2", "HSV tuner", 180, 180, lambda x: None)
    cv.createTrackbar("S lo",  "HSV tuner", 120, 255, lambda x: None)
    cv.createTrackbar("V lo",  "HSV tuner", 70,  255, lambda x: None)
    print("HSV tuner — adjust until mask covers only red blocks, press Q to save")
    while True:
        ret, frame = cap.read()
        if not ret: break
        hsv  = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        h1   = cv.getTrackbarPos("H lo1", "HSV tuner")
        h2   = cv.getTrackbarPos("H hi1", "HSV tuner")
        h3   = cv.getTrackbarPos("H lo2", "HSV tuner")
        h4   = cv.getTrackbarPos("H hi2", "HSV tuner")
        s    = cv.getTrackbarPos("S lo",  "HSV tuner")
        v    = cv.getTrackbarPos("V lo",  "HSV tuner")
        mask = (cv.inRange(hsv, np.array([h1,s,v]), np.array([h2,255,255])) |
                cv.inRange(hsv, np.array([h3,s,v]), np.array([h4,255,255])))
        cv.imshow("HSV tuner", mask)
        cv.imshow("Live feed", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("\n--- Paste into CONFIG ---")
            print(f"RED_LOWER_1 = np.array([{h1}, {s}, {v}])")
            print(f"RED_UPPER_1 = np.array([{h2}, 255, 255])")
            print(f"RED_LOWER_2 = np.array([{h3}, {s}, {v}])")
            print(f"RED_UPPER_2 = np.array([{h4}, 255, 255])")
            break
    cv.destroyAllWindows()

# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────

def main():
    global state

    cap = cv.VideoCapture(CAMERA_INDEX)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  ARENA_W)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, ARENA_H)

    if not cap.isOpened():
        print("ERROR: Cannot open camera. Check CAMERA_INDEX.")
        return

    print("Controls:  SPACE=start  R=reset  S=e-stop  T=HSV tuner  Q=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera feed lost!"); break

        bot_states, trolley_pos, trolley_hdg = detect_markers(frame)
        obstacles                             = detect_obstacles(frame)

        update_state(bot_states, trolley_pos, trolley_hdg, obstacles)
        draw_debug(frame, bot_states, trolley_pos, trolley_hdg, obstacles)

        cv.imshow("Swarm controller", frame)

        key = cv.waitKey(1) & 0xFF
        if   key == ord('q'):
            break
        elif key == ord(' '):
            if state == State.DONE:
                state = State.TRAVEL
            path_cache.clear()
            print("[MANUAL] Run started")
        elif key == ord('r'):
            state = State.TRAVEL
            path_cache.clear()
            print("[MANUAL] Reset to TRAVEL")
        elif key == ord('s'):
            stop_all()
            print("[EMERGENCY STOP]")
        elif key == ord('t'):
            run_hsv_tuner(cap)

    stop_all()
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
