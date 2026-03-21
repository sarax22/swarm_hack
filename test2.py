"""
vision_test.py
──────────────
Standalone test for camera, ArUco detection, red obstacle detection,
and A* pathfinding. No bots, no UDP. Run this before the full controller
to verify everything works with your actual camera and arena.

Controls:
  T  — open HSV tuner to dial in red detection for your lighting
  P  — toggle A* path overlay on/off
  G  — toggle occupancy grid overlay on/off
  R  — force replan all paths now
  Q  — quit
"""

import cv2 as cv
import numpy as np
import heapq
from collections import deque

# ─────────────────────────────────────────────────────────────
# CONFIG — match these to swarm_controller.py on the day
# ─────────────────────────────────────────────────────────────

CAMERA_INDEX  = 1
ARENA_W       = 1280
ARENA_H       = 720

BOT_R_CM      = 4.0
TROLLEY_R_CM  = 4.5
PIXELS_PER_CM = 8.0

BOT_R_PX      = int(BOT_R_CM     * PIXELS_PER_CM)
TROLLEY_R_PX  = int(TROLLEY_R_CM * PIXELS_PER_CM)

CONTACT_PX    = int((BOT_R_CM + TROLLEY_R_CM + 0.5) * PIXELS_PER_CM)
SIDE_Y_PX     = CONTACT_PX

FINISH_LINE_X = ARENA_W - 80
START_LINE_X  = 80
EDGE_MARGIN_PX= BOT_R_PX + 10
OBS_MARGIN_PX = int(8 * PIXELS_PER_CM)
AT_TARGET_PX  = int(3.0 * PIXELS_PER_CM)
MIN_OBS_AREA  = 200

BOT_IDS    = [0, 1, 2]
TROLLEY_ID = 9
DICT_TYPE  = cv.aruco.DICT_4X4_50

# A* grid
ASTAR_CELL_CM = 3
ASTAR_CELL_PX = int(ASTAR_CELL_CM * PIXELS_PER_CM)
GRID_COLS     = ARENA_W // ASTAR_CELL_PX
GRID_ROWS     = ARENA_H // ASTAR_CELL_PX

# Red HSV — tune with T key
RED_LOWER_1 = np.array([0,   120,  70])
RED_UPPER_1 = np.array([10,  255, 255])
RED_LOWER_2 = np.array([170, 120,  70])
RED_UPPER_2 = np.array([180, 255, 255])

# ─────────────────────────────────────────────────────────────
# ARUCO SETUP
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
# RED OBSTACLE DETECTION
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
        # Draw obstacle circle + centre dot
        cv.circle(frame, (int(ox), int(oy)), int(radius), (0, 0, 255), 2)
        cv.circle(frame, (int(ox), int(oy)), 4, (0, 0, 255), -1)
        cv.putText(frame, "OBS",
                   (int(ox) - 15, int(oy) - int(radius) - 6),
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

def build_grid(obstacles, inflation_px, extra_edge_px=0):
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
# FORMATION TARGETS (visual only — no bots commanded)
# ─────────────────────────────────────────────────────────────

def get_push_formation(trolley_pos, push_angle_deg=0.0):
    tx, ty = trolley_pos
    rad    = np.radians(push_angle_deg)
    fwd    = np.array([ np.cos(rad), -np.sin(rad)])
    left   = np.array([-np.sin(rad), -np.cos(rad)])
    return {
        0: (int(tx - fwd[0]*CONTACT_PX),   int(ty - fwd[1]*CONTACT_PX)),   # rear
        1: (int(tx + left[0]*SIDE_Y_PX),   int(ty + left[1]*SIDE_Y_PX)),   # side L
        2: (int(tx - left[0]*SIDE_Y_PX),   int(ty - left[1]*SIDE_Y_PX)),   # side R
    }

# ─────────────────────────────────────────────────────────────
# GRID OVERLAY HELPER
# ─────────────────────────────────────────────────────────────

def draw_grid_overlay(frame, grid):
    """
    Draws the A* occupancy grid on the frame.
    Red cells = blocked, transparent otherwise.
    """
    overlay = frame.copy()
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            if not grid[row, col]:
                x1 = col * ASTAR_CELL_PX
                y1 = row * ASTAR_CELL_PX
                x2 = x1 + ASTAR_CELL_PX
                y2 = y1 + ASTAR_CELL_PX
                cv.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 180), -1)
    cv.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

# ─────────────────────────────────────────────────────────────
# HSV TUNER
# ─────────────────────────────────────────────────────────────

def run_hsv_tuner(cap):
    """
    Live HSV tuner. Adjust sliders until the white mask cleanly covers
    only the red blocks. Press Q to print values to terminal.
    """
    cv.namedWindow("HSV tuner")
    cv.createTrackbar("H lo1", "HSV tuner", 0,   10,  lambda x: None)
    cv.createTrackbar("H hi1", "HSV tuner", 10,  10,  lambda x: None)
    cv.createTrackbar("H lo2", "HSV tuner", 170, 180, lambda x: None)
    cv.createTrackbar("H hi2", "HSV tuner", 180, 180, lambda x: None)
    cv.createTrackbar("S lo",  "HSV tuner", 120, 255, lambda x: None)
    cv.createTrackbar("V lo",  "HSV tuner", 70,  255, lambda x: None)
    print("\n[HSV TUNER] Adjust sliders — mask should be WHITE on red blocks only.")
    print("[HSV TUNER] Press Q to save and return to main view.\n")

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

        # Side-by-side: live frame | mask
        mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        combined = np.hstack([
            cv.resize(frame,    (640, 360)),
            cv.resize(mask_bgr, (640, 360))
        ])
        cv.putText(combined, "Live (left)  |  Red mask (right)",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv.imshow("HSV tuner", combined)

        if cv.waitKey(1) & 0xFF == ord('q'):
            # Update globals
            global RED_LOWER_1, RED_UPPER_1, RED_LOWER_2, RED_UPPER_2
            RED_LOWER_1 = np.array([h1, s, v])
            RED_UPPER_1 = np.array([h2, 255, 255])
            RED_LOWER_2 = np.array([h3, s, v])
            RED_UPPER_2 = np.array([h4, 255, 255])
            print("--- Paste these into CONFIG in both files ---")
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
        print(f"ERROR: Cannot open camera index {CAMERA_INDEX}.")
        print("Try changing CAMERA_INDEX to 1 or 2 if the arena camera isn't default.")
        return

    print("=" * 55)
    print("  Vision test — no bots, no UDP")
    print("=" * 55)
    print("  T — HSV tuner (tune red detection first)")
    print("  P — toggle A* path overlay")
    print("  G — toggle occupancy grid overlay")
    print("  R — force path replan now")
    print("  Q — quit")
    print("=" * 55)
    print(f"  Grid: {GRID_COLS} x {GRID_ROWS} cells  ({ASTAR_CELL_CM}cm per cell)")
    print(f"  Bot radius: {BOT_R_CM}cm = {BOT_R_PX}px")
    print(f"  Trolley radius: {TROLLEY_R_CM}cm = {TROLLEY_R_PX}px")
    print()

    show_paths = True
    show_grid  = False

    # Cached state
    last_obstacles = []
    last_grid      = np.ones((GRID_ROWS, GRID_COLS), dtype=bool)
    bot_paths      = {}    # bot_id -> list of (px, py) waypoints
    trolley_path   = []    # path from trolley to finish (with formation inflation)
    frame_count    = 0
    REPLAN_EVERY   = 12    # replan every N frames (~0.4s at 30fps)

    PATH_COLOURS = {0: (255, 200,   0),
                    1: (  0, 255, 200),
                    2: (200,   0, 255)}
    TROLLEY_PATH_COL = (0, 255, 100)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera feed lost!"); break

        # ── Detection ───────────────────────────────────────────
        bot_states, trolley_pos, trolley_hdg = detect_markers(frame)
        obstacles                             = detect_obstacles(frame)
        frame_count += 1

        # ── Replan paths periodically ───────────────────────────
        if frame_count % REPLAN_EVERY == 0 or not last_obstacles == obstacles:
            last_obstacles = obstacles

            # Solo bot grid (inflated by bot radius only)
            solo_grid = build_grid(obstacles, BOT_R_PX)

            # Formation grid (inflated by bot + trolley radius — used for
            # the trolley path so the whole formation clears obstacles)
            form_grid = build_grid(obstacles, BOT_R_PX + TROLLEY_R_PX,
                                   extra_edge_px=TROLLEY_R_PX)
            last_grid = solo_grid

            # Path for each bot: current pos → its formation slot
            bot_paths = {}
            if trolley_pos:
                formation = get_push_formation(trolley_pos)
                for bot_id, s in bot_states.items():
                    tgt  = formation.get(bot_id)
                    if tgt is None: continue
                    path = astar(solo_grid, s["center"], tgt)
                    if path:
                        bot_paths[bot_id] = path

            # Path for the trolley formation: trolley pos → finish line
            if trolley_pos:
                goal         = (FINISH_LINE_X, trolley_pos[1])
                trolley_path = astar(form_grid, trolley_pos, goal)

        # ── Grid overlay ─────────────────────────────────────────
        if show_grid:
            draw_grid_overlay(frame, last_grid)

        # ── Arena markers ────────────────────────────────────────
        cv.rectangle(frame,
                     (EDGE_MARGIN_PX, EDGE_MARGIN_PX),
                     (ARENA_W - EDGE_MARGIN_PX, ARENA_H - EDGE_MARGIN_PX),
                     (200, 150, 0), 1)
        cv.line(frame, (START_LINE_X,  0), (START_LINE_X,  ARENA_H), (0,255,0),   2)
        cv.line(frame, (FINISH_LINE_X, 0), (FINISH_LINE_X, ARENA_H), (0,100,255), 2)
        cv.putText(frame, "START",  (START_LINE_X+5,  25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),   2)
        cv.putText(frame, "FINISH", (FINISH_LINE_X+5, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,100,255), 2)

        # ── Trolley ──────────────────────────────────────────────
        if trolley_pos:
            cv.circle(frame, trolley_pos, TROLLEY_R_PX, (0, 255, 0), 2)
            hdg_lbl = f"{trolley_hdg:.0f}d" if trolley_hdg is not None else "?"
            cv.putText(frame, f"TROLLEY  hdg={hdg_lbl}",
                       (trolley_pos[0]+TROLLEY_R_PX+4, trolley_pos[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

            # Formation target slots
            formation = get_push_formation(trolley_pos)
            labels    = {0: "REAR", 1: "SIDE-L", 2: "SIDE-R"}
            for bot_id, tgt in formation.items():
                cv.circle(frame, tgt, BOT_R_PX, (0,200,255), 1)
                cv.putText(frame, labels[bot_id],
                           (tgt[0]+BOT_R_PX+2, tgt[1]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.32, (0,200,255), 1)

        # ── A* path overlays ─────────────────────────────────────
        if show_paths:
            # Trolley formation path to finish (thick green)
            for k in range(len(trolley_path)-1):
                cv.line(frame, trolley_path[k], trolley_path[k+1],
                        TROLLEY_PATH_COL, 3)
            if trolley_path:
                cv.circle(frame, trolley_path[-1], 8, TROLLEY_PATH_COL, -1)
                cv.putText(frame, "TROLLEY PATH",
                           (trolley_path[0][0]+6, trolley_path[0][1]-8),
                           cv.FONT_HERSHEY_SIMPLEX, 0.35, TROLLEY_PATH_COL, 1)

            # Individual bot paths to their formation slots
            for bot_id, path in bot_paths.items():
                col = PATH_COLOURS.get(bot_id, (200,200,200))
                for k in range(len(path)-1):
                    cv.line(frame, path[k], path[k+1], col, 2)
                if path:
                    cv.circle(frame, path[-1], 6, col, -1)

        # ── Bot markers ──────────────────────────────────────────
        for bot_id, s in bot_states.items():
            cx_p, cy_p = s["center"]
            hdg        = s["heading"]
            col        = PATH_COLOURS.get(bot_id, (200,200,200))
            # Bot circle
            cv.circle(frame, (cx_p, cy_p), BOT_R_PX, col, 2)
            # Heading arrow
            ex = int(cx_p + (BOT_R_PX+15)*np.cos(np.radians(hdg)))
            ey = int(cy_p - (BOT_R_PX+15)*np.sin(np.radians(hdg)))
            cv.arrowedLine(frame, (cx_p, cy_p), (ex, ey), col, 2)
            # Label
            cv.putText(frame, f"B{bot_id}  {hdg:.0f}d",
                       (cx_p-20, cy_p+BOT_R_PX+14),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # ── HUD ──────────────────────────────────────────────────
        hud_lines = [
            f"Bots visible:      {len(bot_states)} / 3   IDs: {list(bot_states.keys())}",
            f"Trolley visible:   {'YES' if trolley_pos else 'NO'}",
            f"Obstacles:         {len(obstacles)}",
            f"Trolley path wpts: {len(trolley_path)}",
            f"Path overlay [P]:  {'ON' if show_paths else 'OFF'}",
            f"Grid overlay [G]:  {'ON' if show_grid  else 'OFF'}",
            f"Grid: {GRID_COLS}x{GRID_ROWS}  cell={ASTAR_CELL_CM}cm",
        ]
        # Semi-transparent HUD background
        hud_h = len(hud_lines) * 20 + 12
        overlay = frame.copy()
        cv.rectangle(overlay, (8, 8), (420, 8 + hud_h), (0,0,0), -1)
        cv.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        for i, line in enumerate(hud_lines):
            cv.putText(frame, line, (14, 26 + i*20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.42, (220,220,220), 1)

        # ── Show ─────────────────────────────────────────────────
        cv.imshow("Vision test", frame)

        key = cv.waitKey(1) & 0xFF
        if   key == ord('q'):
            break
        elif key == ord('t'):
            run_hsv_tuner(cap)
        elif key == ord('p'):
            show_paths = not show_paths
            print(f"[PATH overlay] {'ON' if show_paths else 'OFF'}")
        elif key == ord('g'):
            show_grid = not show_grid
            print(f"[GRID overlay] {'ON' if show_grid else 'OFF'}")
        elif key == ord('r'):
            frame_count = 0   # force replan next frame
            bot_paths   = {}
            trolley_path= []
            print("[REPLAN] Forcing replan next frame")

    cap.release()
    cv.destroyAllWindows()
    print("Vision test closed.")


if __name__ == "__main__":
    main()