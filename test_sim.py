"""
Test mode: loads a test image, plans approach + push paths,
and simulates bot movement step-by-step with visualisation.

Usage:
    python test_sim.py                      # uses test_course.png
    python test_sim.py my_image.png         # uses custom image

Controls:
    SPACE  = step one navigation tick per bot
    A      = auto-run (continuous stepping)
    P      = switch from APPROACH to PUSH phase
    R      = reset
    Q      = quit
"""

import cv2
import numpy as np
from astar import astar
import math
import sys

# ═══════════════════════════════════════════════════
#                   CONFIG
# ═══════════════════════════════════════════════════

BOTS = {
    'rear':  {'marker_id': 9,  'color': (255, 0, 255)},   # magenta
    'left':  {'marker_id': 2,  'color': (255, 255, 0)},   # cyan
    'right': {'marker_id': 3,  'color': (0, 165, 255)},   # orange
}

TROLLEY_ID = 7
GOAL_MARKER_A = 5
GOAL_MARKER_B = 6

MM_PER_PIXEL = 1750 / 1920
CHUNK = 5

WAYPOINT_THRESHOLD_PX = 30
ANGLE_THRESHOLD_DEG = 10
FORMATION_OFFSET_PX = 80

# Simulated bot speed per tick
SIM_MOVE_PX = 8
SIM_TURN_DEG = 15


# ═══════════════════════════════════════════════════
#              ARUCO DETECTION
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
    states = {}
    if ids is not None:
        for i, mid in enumerate(ids.flatten()):
            pts = corners[i][0]
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            top_mid = (pts[0] + pts[1]) / 2
            center = np.mean(pts, axis=0)
            delta = top_mid - center
            heading = np.degrees(np.arctan2(-delta[1], delta[0]))
            states[mid] = {"center": (cx, cy), "heading": heading}
    return states


# ═══════════════════════════════════════════════════
#              OCCUPANCY GRID
# ═══════════════════════════════════════════════════

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

    for state in bot_states.values():
        mx, my = state["center"]
        cv2.circle(mask, (mx, my), 60, 0, -1)

    ys, xs = np.where(mask == 255)
    for x, y in zip(xs, ys):
        gc = min(x // CHUNK, CHUNK_W - 1)
        gr = min(y // CHUNK, CHUNK_H - 1)
        grid[gr, gc] = 1

    # Trolley obstacle
    if TROLLEY_ID in bot_states:
        tx, ty = bot_states[TROLLEY_ID]["center"]
        tgc, tgr = tx // CHUNK, ty // CHUNK
        rows, cols = grid.shape
        for r in range(max(0, tgr - 6), min(rows, tgr + 7)):
            for c in range(max(0, tgc - 6), min(cols, tgc + 7)):
                if (r - tgr) ** 2 + (c - tgc) ** 2 <= 36:
                    grid[r, c] = 1

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


def clear_cells(grid, pos, radius=3):
    rows, cols = grid.shape
    gr, gc = pos
    for r in range(max(0, gr - radius), min(rows, gr + radius + 1)):
        for c in range(max(0, gc - radius), min(cols, gc + radius + 1)):
            grid[r, c] = 0


def get_formation_goals(bot_states):
    if TROLLEY_ID not in bot_states:
        return None
    tx, ty = bot_states[TROLLEY_ID]["center"]
    h = math.radians(bot_states[TROLLEY_ID]["heading"])
    off = FORMATION_OFFSET_PX
    return {
        'rear':  (int(ty + off * math.sin(h)) // CHUNK,
                  int(tx - off * math.cos(h)) // CHUNK),
        'left':  (int(ty - off * math.cos(h)) // CHUNK,
                  int(tx - off * math.sin(h)) // CHUNK),
        'right': (int(ty + off * math.cos(h)) // CHUNK,
                  int(tx + off * math.sin(h)) // CHUNK),
    }


def plan_approach(bot_states, grid):
    goals = get_formation_goals(bot_states)
    if goals is None:
        return {}, {}
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
    return paths, goals


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
            nr, nc = r - dr * off_cells, c - dc * off_cells
        elif side == 'left':
            nr, nc = r - dc * off_cells, c + dr * off_cells
        elif side == 'right':
            nr, nc = r + dc * off_cells, c - dr * off_cells
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
#              SIMULATED BOTS
# ═══════════════════════════════════════════════════

def wp_to_px(path, idx):
    if idx >= len(path):
        return None
    r, c = path[idx]
    return (c * CHUNK + CHUNK // 2, r * CHUNK + CHUNK // 2)


def sim_navigate_step(sim_bot, target_px):
    """
    Simulate one navigation tick. Mutates sim_bot in place.
    Returns True if target reached.
    """
    bx, by = sim_bot['x'], sim_bot['y']
    heading = sim_bot['heading']
    tx, ty = target_px

    dx = tx - bx
    dy = -(ty - by)
    dist = math.sqrt(dx ** 2 + dy ** 2)

    if dist < WAYPOINT_THRESHOLD_PX:
        return True

    target_angle = math.degrees(math.atan2(dy, dx))
    turn = (target_angle - heading + 180) % 360 - 180

    if abs(turn) > ANGLE_THRESHOLD_DEG:
        # Turn
        step = min(abs(turn), SIM_TURN_DEG)
        sim_bot['heading'] += step if turn > 0 else -step
        sim_bot['last_cmd'] = f"TURN {step:.0f}{'R' if turn > 0 else 'L'}"
    else:
        # Move forward
        move_px = min(dist, SIM_MOVE_PX)
        rad = math.radians(heading)
        sim_bot['x'] += move_px * math.cos(rad)
        sim_bot['y'] -= move_px * math.sin(rad)
        sim_bot['last_cmd'] = f"FWD {move_px * MM_PER_PIXEL:.0f}mm"

    sim_bot['trail'].append((int(sim_bot['x']), int(sim_bot['y'])))
    return False


# ═══════════════════════════════════════════════════
#              DRAWING
# ═══════════════════════════════════════════════════

def draw_grid_overlay(display, grid, alpha=0.3):
    overlay = display.copy()
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                x1, y1 = c * CHUNK, r * CHUNK
                x2, y2 = x1 + CHUNK, y1 + CHUNK
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 100), -1)
    cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)


def draw_path(display, path, color, thickness=2):
    if not path or len(path) < 2:
        return
    for i in range(len(path) - 1):
        p1 = wp_to_px(path, i)
        p2 = wp_to_px(path, i + 1)
        if p1 and p2:
            cv2.line(display, p1, p2, color, thickness)
    # Draw waypoints
    for i in range(len(path)):
        p = wp_to_px(path, i)
        if p:
            cv2.circle(display, p, 4, color, -1)


def draw_sim_bot(display, sim_bot, color, label):
    x, y = int(sim_bot['x']), int(sim_bot['y'])
    heading = sim_bot['heading']

    # Trail
    if len(sim_bot['trail']) > 1:
        for i in range(1, len(sim_bot['trail'])):
            cv2.line(display, sim_bot['trail'][i - 1], sim_bot['trail'][i],
                     color, 1)

    # Body
    cv2.circle(display, (x, y), 12, color, 2)

    # Heading arrow
    rad = math.radians(heading)
    ax = int(x + 18 * math.cos(rad))
    ay = int(y - 18 * math.sin(rad))
    cv2.arrowedLine(display, (x, y), (ax, ay), color, 2, tipLength=0.4)

    # Label
    cv2.putText(display, f"{label}: {sim_bot.get('last_cmd', 'idle')}",
                (x + 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def draw_formation_goals(display, goals):
    if goals is None:
        return
    labels = {'rear': 'R', 'left': 'L', 'right': 'Ri'}
    for role, (gr, gc) in goals.items():
        px = gc * CHUNK + CHUNK // 2
        py = gr * CHUNK + CHUNK // 2
        cv2.drawMarker(display, (px, py), (0, 255, 255),
                       cv2.MARKER_CROSS, 15, 2)
        cv2.putText(display, labels.get(role, role), (px + 8, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)


# ═══════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════

STATE_APPROACH = 'APPROACH'
STATE_PUSH = 'PUSH'
STATE_DONE = 'DONE'

if __name__ == "__main__":

    # Load test image
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test_course.png"
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Could not load {img_path}")
        sys.exit(1)
    print(f"Loaded: {img_path} ({frame.shape[1]}x{frame.shape[0]})")

    # Detect markers
    bot_states = detect_markers(frame)
    print(f"Detected markers: {list(bot_states.keys())}")
    for mid, s in bot_states.items():
        print(f"  ID {mid}: center={s['center']}, heading={s['heading']:.1f}")

    # Build grid
    grid = build_occupancy_grid(frame, bot_states)
    print(f"Grid: {grid.shape} ({np.sum(grid)} occupied cells)")

    # Plan approach paths
    approach_paths, formation_goals = plan_approach(bot_states, grid.copy())
    print(f"\nApproach paths planned: {list(approach_paths.keys())}")
    for role, p in approach_paths.items():
        print(f"  {role}: {len(p)} waypoints")

    # Plan push paths
    push_grid = grid.copy()
    if TROLLEY_ID in bot_states:
        tx, ty = bot_states[TROLLEY_ID]["center"]
        clear_cells(push_grid, (ty // CHUNK, tx // CHUNK), radius=8)
    push_paths = plan_push(bot_states, push_grid)
    print(f"\nPush paths planned: {list(push_paths.keys())}")
    for role, p in push_paths.items():
        print(f"  {role}: {len(p)} waypoints")

    goal_pos = get_goal_px(bot_states)
    print(f"\nGoal position: {goal_pos}")

    # Initialise simulated bots from marker positions
    sim_bots = {}
    for role, info in BOTS.items():
        mid = info['marker_id']
        if mid in bot_states:
            bx, by = bot_states[mid]["center"]
            sim_bots[role] = {
                'x': float(bx),
                'y': float(by),
                'heading': bot_states[mid]["heading"],
                'trail': [(bx, by)],
                'last_cmd': 'idle',
            }
            print(f"  {role} bot (marker {mid}): ({bx}, {by}), {sim_bots[role]['heading']:.1f} deg")

    # State
    state = STATE_APPROACH
    wp_idx = {role: 0 for role in approach_paths}
    auto_run = False
    tick = 0

    print(f"\n--- Simulation ready ---")
    print(f"SPACE=step  A=auto  P=push  R=reset  Q=quit\n")

    while True:
        # Draw
        display = frame.copy()
        draw_grid_overlay(display, grid, alpha=0.2)

        # Draw goal
        if goal_pos:
            cv2.circle(display, goal_pos, 12, (0, 0, 255), 2)
            cv2.putText(display, "GOAL", (goal_pos[0] + 14, goal_pos[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw trolley
        if TROLLEY_ID in bot_states:
            tx, ty = bot_states[TROLLEY_ID]["center"]
            cv2.circle(display, (tx, ty), 15, (255, 255, 255), 2)
            cv2.putText(display, "TROLLEY", (tx + 18, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw paths
        if state == STATE_APPROACH:
            draw_formation_goals(display, formation_goals)
            for role, p in approach_paths.items():
                draw_path(display, p, BOTS[role]['color'])
        elif state == STATE_PUSH:
            for role, p in push_paths.items():
                draw_path(display, p, BOTS[role]['color'])

        # Draw simulated bots
        for role, sb in sim_bots.items():
            draw_sim_bot(display, sb, BOTS[role]['color'], role)

        # Status bar
        cv2.putText(display, f"State: {state} | Tick: {tick} | Auto: {'ON' if auto_run else 'OFF'}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        # Waypoint progress
        y_off = 45
        for role in BOTS:
            paths_ref = approach_paths if state == STATE_APPROACH else push_paths
            if role in paths_ref:
                idx = wp_idx.get(role, 0)
                total = len(paths_ref[role])
                cv2.putText(display, f"{role}: wp {min(idx+1, total)}/{total}",
                            (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            BOTS[role]['color'], 1)
                y_off += 18

        cv2.imshow("Simulation", display)

        # Input
        wait_time = 50 if auto_run else 0
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('a'):
            auto_run = not auto_run
            print(f"Auto-run: {'ON' if auto_run else 'OFF'}")
        elif key == ord('r'):
            # Reset sim bots to original positions
            state = STATE_APPROACH
            wp_idx = {role: 0 for role in approach_paths}
            tick = 0
            auto_run = False
            for role, info in BOTS.items():
                mid = info['marker_id']
                if mid in bot_states:
                    bx, by = bot_states[mid]["center"]
                    sim_bots[role] = {
                        'x': float(bx), 'y': float(by),
                        'heading': bot_states[mid]["heading"],
                        'trail': [(bx, by)], 'last_cmd': 'idle',
                    }
            print("Reset!")
            continue
        elif key == ord('p'):
            print("\n=== Switching to PUSH phase ===\n")
            state = STATE_PUSH
            wp_idx = {role: 0 for role in push_paths}
            continue
        elif key != ord(' ') and not auto_run:
            continue

        # ─── SIMULATION TICK ──────────────────────

        if state == STATE_DONE:
            continue

        paths_ref = approach_paths if state == STATE_APPROACH else push_paths
        all_done = True

        for role, info in BOTS.items():
            if role not in paths_ref or role not in sim_bots:
                continue

            p = paths_ref[role]
            idx = wp_idx.get(role, 0)

            if idx >= len(p):
                continue

            all_done = False
            target = wp_to_px(p, idx)
            if target is None:
                continue

            reached = sim_navigate_step(sim_bots[role], target)
            if reached:
                wp_idx[role] = idx + 1
                if idx + 1 >= len(p):
                    print(f"  [{role}] ARRIVED at final waypoint!")
                else:
                    print(f"  [{role}] reached wp {idx + 1}/{len(p)}")

        tick += 1

        if all_done:
            if state == STATE_APPROACH:
                print("\n=== All bots in formation! Press P for PUSH ===\n")
                state = STATE_PUSH
                wp_idx = {role: 0 for role in push_paths}
            elif state == STATE_PUSH:
                print("\n=== PUSH COMPLETE! ===\n")
                state = STATE_DONE
                auto_run = False

    cv2.destroyAllWindows()
    print("Done.")