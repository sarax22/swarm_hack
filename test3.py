"""
single_bot_test.py
──────────────────
Test one MONA bot navigating from start line to finish line
using A* pathfinding, avoiding red obstacles.
No trolley, no formation, no swarm logic.

Controls:
  SPACE — send bot to finish line
  R     — reset / send bot back to start
  S     — emergency stop
  T     — HSV tuner for red obstacles
  Q     — quit
"""

import cv2 as cv
import numpy as np
import socket
import heapq
import time
from collections import deque

# ─────────────────────────────────────────────────────────────
# CONFIG — update these on the day
# ─────────────────────────────────────────────────────────────

MONA_IP      = "192.168.0.120"   # IP of the single bot being tested
BOT_ID       = 9                  # ArUco marker ID on this bot
UDP_PORT     = 80
CAMERA_INDEX = 0

ARENA_W = 1280
ARENA_H = 720

BOT_R_CM      = 4.0
PIXELS_PER_CM = 8.0
BOT_R_PX      = int(BOT_R_CM * PIXELS_PER_CM)

FINISH_LINE_X  = ARENA_W - 80
START_LINE_X   = 80
EDGE_MARGIN_PX = BOT_R_PX + 10
OBS_MARGIN_PX  = int(8 * PIXELS_PER_CM)
MIN_OBS_AREA   = 200

AT_TARGET_PX   = int(3.0 * PIXELS_PER_CM)   # 3cm "arrived"
BASE_SPEED     = 75
TURN_GAIN      = 1.5
REPLAN_EVERY   = 12    # frames between A* replans (~0.4s at 30fps)

DICT_TYPE = cv.aruco.DICT_4X4_50

# Red HSV — tune with T key first
RED_LOWER_1 = np.array([0,   120,  70])
RED_UPPER_1 = np.array([10,  255, 255])
RED_LOWER_2 = np.array([170, 120,  70])
RED_UPPER_2 = np.array([180, 255, 255])

# A* grid
ASTAR_CELL_CM = 3
ASTAR_CELL_PX = int(ASTAR_CELL_CM * PIXELS_PER_CM)
GRID_COLS     = ARENA_W // ASTAR_CELL_PX
GRID_ROWS     = ARENA_H // ASTAR_CELL_PX

# ─────────────────────────────────────────────────────────────
# UDP — single bot only
# ─────────────────────────────────────────────────────────────

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_command(left: int, right: int):
    left  = int(np.clip(left,  -100, 100))
    right = int(np.clip(right, -100, 100))
    msg   = f"{BOT_ID},{left},{right}".encode()
    sock.sendto(msg, (MONA_IP, UDP_PORT))

def stop():
    send_command(0, 0)

# ─────────────────────────────────────────────────────────────
# VISION — ArUco (single bot)
# ─────────────────────────────────────────────────────────────

aruco_dict   = cv.aruco.getPredefinedDictionary(DICT_TYPE)
aruco_params = cv.aruco.DetectorParameters()
detector     = cv.aruco.ArucoDetector(aruco_dict, aruco_params)

def detect_bot(frame):
    """
    Returns (center, heading) for BOT_ID or (None, None) if not visible.
    center  = (px, py)
    heading = degrees, 0=right, 90=up, 180=left
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, None
    cv.aruco.drawDetectedMarkers(frame, corners, ids)
    for i, mid in enumerate(ids.flatten()):
        if mid != BOT_ID:
            continue
        pts     = corners[i][0]
        cx      = int(np.mean(pts[:, 0]))
        cy      = int(np.mean(pts[:, 1]))
        top_mid = (pts[0] + pts[1]) / 2.0
        center  = np.mean(pts, axis=0)
        delta   = top_mid - center
        heading = float(np.degrees(np.arctan2(-delta[1], delta[0])))
        return (cx, cy), heading
    return None, None

# ─────────────────────────────────────────────────────────────
# VISION — Red obstacles
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
        obstacles.append((int(ox), int(oy), int(radius)))
        cv.circle(frame, (int(ox), int(oy)), int(radius), (0, 0, 255), 2)
        cv.circle(frame, (int(ox), int(oy)), 4,            (0, 0, 255), -1)
        cv.putText(frame, "OBS",
                   (int(ox)-15, int(oy)-int(radius)-5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return obstacles

# ─────────────────────────────────────────────────────────────
# A* PATHFINDING
# ─────────────────────────────────────────────────────────────

def px_to_cell(px, py):
    return (max(0, min(GRID_COLS-1, int(px // ASTAR_CELL_PX))),
            max(0, min(GRID_ROWS-1, int(py // ASTAR_CELL_PX))))

def cell_to_px(col, row):
    return (int((col + 0.5) * ASTAR_CELL_PX),
            int((row + 0.5) * ASTAR_CELL_PX))

def build_grid(obstacles):
    """
    Builds bool grid inflated by BOT_R_PX so the bot centre can be
    treated as a point — the real circle never clips an obstacle.
    """
    grid = np.ones((GRID_ROWS, GRID_COLS), dtype=bool)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            cx_c, cy_c = cell_to_px(col, row)
            # Arena edge keep-out
            if (cx_c < EDGE_MARGIN_PX or cx_c > ARENA_W - EDGE_MARGIN_PX or
                    cy_c < EDGE_MARGIN_PX or cy_c > ARENA_H - EDGE_MARGIN_PX):
                grid[row, col] = False
                continue
            # Obstacle inflation
            for (ox, oy, obs_r) in obstacles:
                if np.hypot(cx_c - ox, cy_c - oy) < obs_r + BOT_R_PX:
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
    c1, r1 = px_to_cell(*p1); c2, r2 = px_to_cell(*p2)
    dc = abs(c2-c1); dr = abs(r2-r1)
    sc = 1 if c1 < c2 else -1; sr = 1 if r1 < r2 else -1
    err = dc - dr; c, r = c1, r1
    while True:
        if not (0 <= c < GRID_COLS and 0 <= r < GRID_ROWS): return False
        if not grid[r, c]: return False
        if c == c2 and r == r2: return True
        e2 = 2*err
        if e2 > -dr: err -= dr; c += sc
        if e2 <  dc: err += dc; r += sr

def smooth_path(path, grid):
    if len(path) <= 2: return path
    smoothed = [path[0]]; i = 0
    while i < len(path)-1:
        j = len(path)-1
        while j > i+1:
            if line_of_sight(grid, path[i], path[j]): break
            j -= 1
        smoothed.append(path[j]); i = j
    return smoothed

def astar(grid, start_px, goal_px):
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
# CONTROL — steering along waypoints
# ─────────────────────────────────────────────────────────────

def steer_to_waypoint(pos, heading, wp):
    dx      = wp[0] - pos[0]
    dy      = wp[1] - pos[1]
    desired = np.degrees(np.arctan2(-dy, dx))
    error   = (desired - heading + 180) % 360 - 180
    corr    = int(error * TURN_GAIN)

    # Slow down when the heading error is large — turn on the spot first
    if abs(error) > 45:
        # Spin in place to face the waypoint before driving forward
        if error > 0:
            send_command(-40, 40)   # turn left
        else:
            send_command(40, -40)   # turn right
        return

    spd   = max(40, int(BASE_SPEED * min(1.0, np.hypot(dx, dy) / (AT_TARGET_PX * 4))))
    left  = int(np.clip(spd - corr, -100, 100))
    right = int(np.clip(spd + corr, -100, 100))
    send_command(left, right)

# ─────────────────────────────────────────────────────────────
# GRID OVERLAY
# ─────────────────────────────────────────────────────────────

def draw_grid_overlay(frame, grid):
    overlay = frame.copy()
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            if not grid[row, col]:
                x1 = col * ASTAR_CELL_PX; y1 = row * ASTAR_CELL_PX
                cv.rectangle(overlay, (x1,y1),
                             (x1+ASTAR_CELL_PX, y1+ASTAR_CELL_PX),
                             (0, 0, 180), -1)
    cv.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

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
    print("\n[HSV TUNER] Mask should be WHITE on red blocks only. Q to save.\n")
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
        combined = np.hstack([cv.resize(frame,                     (640,360)),
                               cv.resize(cv.cvtColor(mask, cv.COLOR_GRAY2BGR), (640,360))])
        cv.putText(combined, "Live  |  Red mask",
                   (10,28), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv.imshow("HSV tuner", combined)
        if cv.waitKey(1) & 0xFF == ord('q'):
            global RED_LOWER_1, RED_UPPER_1, RED_LOWER_2, RED_UPPER_2
            RED_LOWER_1 = np.array([h1, s, v])
            RED_UPPER_1 = np.array([h2, 255, 255])
            RED_LOWER_2 = np.array([h3, s, v])
            RED_UPPER_2 = np.array([h4, 255, 255])
            print("--- Paste into CONFIG ---")
            print(f"RED_LOWER_1 = np.array([{h1}, {s}, {v}])")
            print(f"RED_UPPER_1 = np.array([{h2}, 255, 255])")
            print(f"RED_LOWER_2 = np.array([{h3}, {s}, {v}])")
            print(f"RED_UPPER_2 = np.array([{h4}, 255, 255])")
            break
    cv.destroyWindow("HSV tuner")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    cap = cv.VideoCapture(CAMERA_INDEX)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  ARENA_W)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, ARENA_H)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_INDEX}")
        return

    print("=" * 50)
    print("  Single bot test — no trolley")
    print("=" * 50)
    print(f"  Bot ID:  {BOT_ID}")
    print(f"  Bot IP:  {MONA_IP}")
    print()
    print("  SPACE — go to finish")
    print("  R     — go back to start")
    print("  S     — emergency stop")
    print("  T     — HSV tuner")
    print("  G     — toggle grid overlay")
    print("  Q     — quit")
    print("=" * 50)

    running     = False      # True = bot is navigating
    goal        = (FINISH_LINE_X, ARENA_H // 2)
    path        = []
    waypoint_idx= 0
    frame_count = 0
    show_grid   = False
    last_grid   = np.ones((GRID_ROWS, GRID_COLS), dtype=bool)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera lost!"); break

        bot_pos, bot_hdg = detect_bot(frame)
        obstacles        = detect_obstacles(frame)
        frame_count     += 1

        # ── Replan A* periodically or when obstacles change ──────
        if running and bot_pos and frame_count % REPLAN_EVERY == 0:
            last_grid = build_grid(obstacles)
            new_path  = astar(last_grid, bot_pos, goal)
            if new_path:
                # Find closest waypoint on new path to avoid snapping backward
                path         = new_path
                waypoint_idx = 0

        # ── Navigation ───────────────────────────────────────────
        if running and bot_pos and path:
            # Advance waypoint index when close enough
            while (waypoint_idx < len(path) - 1 and
                   np.hypot(bot_pos[0]-path[waypoint_idx][0],
                            bot_pos[1]-path[waypoint_idx][1]) < AT_TARGET_PX):
                waypoint_idx += 1

            wp = path[waypoint_idx]
            steer_to_waypoint(bot_pos, bot_hdg, wp)

            # Check if reached goal
            if np.hypot(bot_pos[0]-goal[0], bot_pos[1]-goal[1]) < AT_TARGET_PX * 2:
                print("[DONE] Bot reached goal!")
                stop()
                running = False
                path    = []

        elif running and bot_pos is None:
            # Lost the bot marker — stop and wait
            stop()

        # ── Grid overlay ─────────────────────────────────────────
        if show_grid:
            draw_grid_overlay(frame, last_grid)

        # ── Arena lines ──────────────────────────────────────────
        cv.rectangle(frame,
                     (EDGE_MARGIN_PX, EDGE_MARGIN_PX),
                     (ARENA_W-EDGE_MARGIN_PX, ARENA_H-EDGE_MARGIN_PX),
                     (200,150,0), 1)
        cv.line(frame, (START_LINE_X,  0), (START_LINE_X,  ARENA_H), (0,255,0),   2)
        cv.line(frame, (FINISH_LINE_X, 0), (FINISH_LINE_X, ARENA_H), (0,100,255), 2)
        cv.putText(frame, "START",  (START_LINE_X+5,  25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),   2)
        cv.putText(frame, "FINISH", (FINISH_LINE_X+5, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,100,255), 2)

        # ── Draw A* path ─────────────────────────────────────────
        for k in range(len(path)-1):
            colour = (0,255,200) if k >= waypoint_idx else (80,80,80)
            cv.line(frame, path[k], path[k+1], colour, 2)
        if path and waypoint_idx < len(path):
            cv.circle(frame, path[waypoint_idx], 8, (0,255,200), -1)
            cv.putText(frame, "WP",
                       (path[waypoint_idx][0]+8, path[waypoint_idx][1]-8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,200), 1)

        # Goal marker
        cv.circle(frame, goal, 12, (0,100,255), -1)
        cv.putText(frame, "GOAL", (goal[0]+10, goal[1]-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (0,100,255), 1)

        # ── Draw bot ─────────────────────────────────────────────
        if bot_pos:
            cv.circle(frame, bot_pos, BOT_R_PX, (255,200,0), 2)
            if bot_hdg is not None:
                ex = int(bot_pos[0] + (BOT_R_PX+15)*np.cos(np.radians(bot_hdg)))
                ey = int(bot_pos[1] - (BOT_R_PX+15)*np.sin(np.radians(bot_hdg)))
                cv.arrowedLine(frame, bot_pos, (ex,ey), (255,200,0), 2)
            cv.putText(frame, f"B{BOT_ID}  {bot_hdg:.0f}d" if bot_hdg else f"B{BOT_ID}",
                       (bot_pos[0]-20, bot_pos[1]+BOT_R_PX+14),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        else:
            cv.putText(frame, f"BOT {BOT_ID} NOT VISIBLE",
                       (ARENA_W//2 - 120, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # ── HUD ──────────────────────────────────────────────────
        status   = "RUNNING" if running else "STOPPED"
        obs_str  = str(len(obstacles))
        wps_str  = f"{waypoint_idx}/{len(path)}" if path else "—"
        bot_str  = f"{bot_pos}" if bot_pos else "NOT VISIBLE"
        hud = [
            f"Status:     {status}",
            f"Bot {BOT_ID}:     {bot_str}",
            f"Obstacles:  {obs_str}",
            f"Waypoint:   {wps_str}",
            f"Grid [G]:   {'ON' if show_grid else 'OFF'}",
        ]
        ov = frame.copy()
        cv.rectangle(ov, (8,8), (310, 8+len(hud)*22+8), (0,0,0), -1)
        cv.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
        for i, line in enumerate(hud):
            col = (0,200,100) if "RUNNING" in line else (220,220,220)
            cv.putText(frame, line, (14, 28+i*22),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        cv.imshow("Single bot test", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if bot_pos is None:
                print("[WARN] Bot not visible — cannot start")
            else:
                goal        = (FINISH_LINE_X, ARENA_H // 2)
                last_grid   = build_grid(obstacles)
                path        = astar(last_grid, bot_pos, goal)
                waypoint_idx= 0
                running     = True
                frame_count = 0
                print(f"[GO] Bot → finish  path={len(path)} waypoints")
        elif key == ord('r'):
            if bot_pos is None:
                print("[WARN] Bot not visible — cannot navigate")
            else:
                goal        = (START_LINE_X, ARENA_H // 2)
                last_grid   = build_grid(obstacles)
                path        = astar(last_grid, bot_pos, goal)
                waypoint_idx= 0
                running     = True
                frame_count = 0
                print(f"[RETURN] Bot → start  path={len(path)} waypoints")
        elif key == ord('s'):
            stop()
            running = False
            path    = []
            print("[EMERGENCY STOP]")
        elif key == ord('t'):
            stop()
            run_hsv_tuner(cap)
        elif key == ord('g'):
            show_grid = not show_grid
            if show_grid:
                last_grid = build_grid(obstacles)

    stop()
    cap.release()
    cv.destroyAllWindows()
    print("Test closed.")


if __name__ == "__main__":
    main()