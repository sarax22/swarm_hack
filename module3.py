import cv2
import numpy as np
from astar import astar
import time
import matplotlib.pyplot as plt
import socket
import math
 
# ═══════════════════════════════════════════════════
#              ROBOT WiFi CONNECTION
# ═══════════════════════════════════════════════════
ROBOT_HOST = '192.168.0.120'
ROBOT_PORT = 80
 
def connect_robot(host, port, timeout=20):
    """Connect to the Mona robot over WiFi and return the socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((host, port))
    print(f"Connected to robot at {host}:{port}")
    return sock
 
def recv_line(sock):
    """Read exactly one line from the socket — guarantees clean handshake."""
    data = b''
    while True:
        byte = sock.recv(1)
        if not byte:
            raise ConnectionError("Socket closed by robot")
        if byte == b'\n':
            return data.decode().strip()
        data += byte
 
def send_command(sock, cmd, value):
    """
    Send a single command to the robot and wait for OK.
    cmd: 'F', 'B', 'R', 'L', 'S'
    value: int (mm for F/B, degrees for R/L, ignored for S)
    Returns the robot's response string.
    """
    packet = f"{cmd}{value}\n"
    print(f"  -> Sending: {packet.strip()} ... ", end="", flush=True)
    sock.sendall(packet.encode())
    response = recv_line(sock)
    print(f"Robot: {response}")
    return response
 
def execute_route(sock, commands):
    """
    Execute a list of (type, value) commands on the robot with handshaking.
    Each command waits for 'OK' before sending the next.
    
    commands: list of ('MOVE', mm) or ('ROTATE', degrees) tuples
    """
    print(f"\n--- Executing route: {len(commands)} steps ---")
 
    for i, (cmd_type, value) in enumerate(commands):
        print(f"[{i+1}/{len(commands)}] ", end="")
 
        if cmd_type == 'MOVE':
            robot_cmd = 'F'
            int_val = round(abs(value))
        elif cmd_type == 'ROTATE':
            # Positive = right turn, Negative = left turn
            robot_cmd = 'R' if value > 0 else 'L'
            int_val = round(abs(value))
        else:
            print(f"Unknown command type: {cmd_type}, skipping")
            continue
 
        if int_val == 0:
            print(f"Skipping zero-value {cmd_type}")
            continue
 
        try:
            response = send_command(sock, robot_cmd, int_val)
            if response != "OK":
                print(f"WARNING: Expected 'OK', got '{response}'")
        except socket.timeout:
            print("TIMEOUT — robot not responding, sending stop")
            try:
                send_command(sock, 'S', 0)
            except:
                pass
            return False
 
    print("--- Route complete! ---\n")
    return True
 
 
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
 
BOT_NAMES = {0: "Bot_A", 1: "Bot_B", 2: "Bot_C"}
 
 
def get_marker_center(corners_single):
    pts = corners_single[0]
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    return cx, cy
 
def get_marker_heading(corners_single):
    pts = corners_single[0]
    top_mid = ((pts[0] + pts[1]) / 2)
    center  = np.mean(pts, axis=0)
    delta   = top_mid - center
    angle   = np.degrees(np.arctan2(-delta[1], delta[0]))
    return angle
 
 
# ═══════════════════════════════════════════════════
#              PATH PROCESSING
# ═══════════════════════════════════════════════════
 
def simplify_path(path):
    """Remove intermediate points on straight segments."""
    if not path or len(path) < 3:
        return path
    simplified = [path[0]]
    for i in range(1, len(path) - 1):
        dy1 = path[i][0] - path[i-1][0]
        dx1 = path[i][1] - path[i-1][1]
        dy2 = path[i+1][0] - path[i][0]
        dx2 = path[i+1][1] - path[i][1]
        if (dx1, dy1) != (dx2, dy2):
            simplified.append(path[i])
    simplified.append(path[-1])
    return simplified
 
 
def path_to_commands(simplified_path, current_heading_deg, cell_size=5, mm_per_pixel=1750/1920):
    """
    Convert simplified A* path to ROTATE and MOVE commands.
    
    Args:
        simplified_path: list of (row, col) waypoints
        current_heading_deg: bot's current heading from ArUco
        cell_size: pixels per grid cell
        mm_per_pixel: physical scale
    
    Returns:
        list of ('ROTATE', angle_deg) and ('MOVE', distance_mm) tuples
    """
    if not simplified_path or len(simplified_path) < 2:
        return []
 
    commands = []
    heading = current_heading_deg
 
    for i in range(1, len(simplified_path)):
        r0, c0 = simplified_path[i - 1]
        r1, c1 = simplified_path[i]
 
        dx = (c1 - c0) * cell_size
        dy = -(r1 - r0) * cell_size  # row increases downward
 
        target_angle = math.degrees(math.atan2(dy, dx))
 
        # Shortest turn
        turn = (target_angle - heading + 180) % 360 - 180
 
        if abs(turn) > 1:  # dead zone
            commands.append(('ROTATE', round(turn, 1)))
 
        dist_px = math.sqrt(dx**2 + dy**2)
        dist_mm = dist_px * mm_per_pixel
        commands.append(('MOVE', round(dist_mm, 1)))
 
        heading = target_angle
 
    return commands

def get_formation_goals(bot_states, trolley_id=7, offset_px=80, chunk=5):
    """
    Compute three goal positions around the trolley:
    - rear: directly behind (opposite to trolley heading)
    - left: to the left side
    - right: to the right side
    
    Args:
        bot_states: dict from process_frame
        trolley_id: marker ID of the trolley
        offset_px: how far from trolley center each bot should target (in pixels)
        chunk: grid cell size
    
    Returns:
        (rear_goal, left_goal, right_goal) as grid coords (row, col)
        or None if trolley not detected
    """
    if trolley_id not in bot_states:
        print("Trolley not detected!")
        return None

    tx, ty = bot_states[trolley_id]["center"]
    heading_deg = bot_states[trolley_id]["heading"]
    heading_rad = math.radians(heading_deg)

    # Rear: opposite to trolley's heading direction
    rear_x = tx - offset_px * math.cos(heading_rad)
    rear_y = ty + offset_px * math.sin(heading_rad)  # +y because screen coords

    # Left: 90 degrees left of heading
    left_x = tx - offset_px * math.sin(heading_rad)
    left_y = ty - offset_px * math.cos(heading_rad)

    # Right: 90 degrees right of heading
    right_x = tx + offset_px * math.sin(heading_rad)
    right_y = ty + offset_px * math.cos(heading_rad)

    # Convert to grid coords (row, col)
    rear_goal  = (int(rear_y) // chunk, int(rear_x) // chunk)
    left_goal  = (int(left_y) // chunk, int(left_x) // chunk)
    right_goal = (int(right_y) // chunk, int(right_x) // chunk)

    return rear_goal, left_goal, right_goal

def clear_goal_cells(grid, goal, radius=2):
    """Clear a small area around a goal so A* can actually reach it."""
    rows, cols = grid.shape
    gr, gc = goal
    for r in range(max(0, gr - radius), min(rows, gr + radius + 1)):
        for c in range(max(0, gc - radius), min(cols, gc + radius + 1)):
            grid[r, c] = 0

def get_push_path(bot_states, inflated_grid, trolley_id=7, 
                  goal_marker_a=5, goal_marker_b=6, chunk=5):
    """
    Compute a path from the trolley to the midpoint between two goal markers.
    Returns the simplified path in grid coords, or None.
    """
    if trolley_id not in bot_states:
        print("Trolley not detected!")
        return None
    if goal_marker_a not in bot_states or goal_marker_b not in bot_states:
        print("Goal markers not detected!")
        return None

    # Goal = midpoint between the two markers
    ax, ay = bot_states[goal_marker_a]["center"]
    bx, by = bot_states[goal_marker_b]["center"]
    goal_x = (ax + bx) // 2
    goal_y = (ay + by) // 2

    # Trolley start
    tx, ty = bot_states[trolley_id]["center"]

    start = (ty // chunk, tx // chunk)
    goal = (goal_y // chunk, goal_x // chunk)

    # Clear start and goal on grid
    clear_goal_cells(inflated_grid, start, radius=3)
    clear_goal_cells(inflated_grid, goal, radius=3)

    raw_path = astar(inflated_grid, start, goal)
    if raw_path is None:
        print("No path found for trolley!")
        return None

    return simplify_path(raw_path)

def offset_path(path, bot_states, trolley_id=7, offset_px=80, 
                side='rear', chunk=5):
    """
    Offset a trolley path for a specific bot position.
    
    Args:
        path: simplified path in grid coords (row, col)
        bot_states: current detection state
        trolley_id: trolley marker ID
        offset_px: distance from trolley center in pixels
        side: 'rear', 'left', or 'right'
        chunk: grid cell size
    
    Returns:
        offset path as list of (row, col) grid coords
    """
    if not path or len(path) < 2:
        return path

    offset_cells = offset_px / chunk
    offset_path_out = []

    for i in range(len(path)):
        r, c = path[i]

        # Compute direction of travel at this point
        if i < len(path) - 1:
            dr = path[i + 1][0] - r
            dc = path[i + 1][1] - c
        else:
            dr = r - path[i - 1][0]
            dc = c - path[i - 1][1]

        # Normalise
        length = math.sqrt(dr**2 + dc**2)
        if length == 0:
            offset_path_out.append((r, c))
            continue
        dr /= length
        dc /= length

        if side == 'rear':
            # Behind the trolley (opposite to direction of travel)
            new_r = r - dr * offset_cells
            new_c = c - dc * offset_cells
        elif side == 'left':
            # Perpendicular left (rotate direction 90° CCW)
            new_r = r - dc * offset_cells
            new_c = c + dr * offset_cells
        elif side == 'right':
            # Perpendicular right (rotate direction 90° CW)
            new_r = r + dc * offset_cells
            new_c = c - dr * offset_cells
        else:
            new_r, new_c = r, c

        offset_path_out.append((round(new_r), round(new_c)))

    return offset_path_out
 
 
# ═══════════════════════════════════════════════════
#              FRAME PROCESSING
# ═══════════════════════════════════════════════════
 
# green = 0, 11
# trolley = 19
#  

def process_frame(frame):
    grid = 0
    path1 = path2 = path3 = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
 
    bot_states = {}
 
    if ids is not None:
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
 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
 
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2
 
        edges = cv2.Canny(red_mask, 50, 150)
 
        cv2.imshow("EDGE MASK", edges)
        cv2.waitKey(0)
 
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
        HEIGHT, WIDTH = edges.shape
        CHUNK = 5
        CHUNK_H = HEIGHT // CHUNK
        CHUNK_W = WIDTH // CHUNK
 
        grid = np.zeros((CHUNK_H, CHUNK_W), dtype=int)
 
        mask = np.zeros_like(edges)
 
        cv2.imshow("RED MASK", red_mask)
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
            cx = min(x // CHUNK, CHUNK_W - 1)
            cy = min(y // CHUNK, CHUNK_H - 1)
            grid[cy, cx] = 1
 
        inflate_radius = 3
        grid_img = (grid.astype(np.uint8)) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflate_radius*2+1, inflate_radius*2+1))
        inflated = cv2.dilate(grid_img, kernel)
        inflated_grid = (inflated > 0).astype(int)
 
        # start1 = (bot_states[4]["center"][1]//CHUNK, bot_states[4]["center"][0]//CHUNK)
        # start2 = (bot_states[2]["center"][1]//CHUNK, bot_states[2]["center"][0]//CHUNK)
        start3 = (bot_states[9]["center"][1]//CHUNK, bot_states[3]["center"][0]//CHUNK)
        goals = get_formation_goals(bot_states, trolley_id=19, offset_px=80, chunk=CHUNK)
        
        if goals:
            rear_goal, left_goal, right_goal = goals

            clear_goal_cells(inflated_grid, rear_goal)
            clear_goal_cells(inflated_grid, left_goal)
            clear_goal_cells(inflated_grid, right_goal)


            # start1 = (bot_states[4]["center"][1]//CHUNK, bot_states[4]["center"][0]//CHUNK)
            # start2 = (bot_states[2]["center"][1]//CHUNK, bot_states[2]["center"][0]//CHUNK)
            start3 = (bot_states[9]["center"][1]//CHUNK, bot_states[9]["center"][0]//CHUNK)

            # Bot 4 = rear pusher, Bot 2 = left, Bot 3 = right
            # path1 = astar(inflated_grid, start1, rear_goal)
            # path2 = astar(inflated_grid, start2, left_goal)
            path3 = astar(inflated_grid, start3, right_goal)
 
    return frame, bot_states, grid, 0, 0, path3
 
 
# ═══════════════════════════════════════════════════
#                    MAIN
# ═══════════════════════════════════════════════════
 
if __name__ == "__main__":
 
    # --- Connect to robot ---
    robot_sock = None
    try:
        robot_sock = connect_robot(ROBOT_HOST, ROBOT_PORT)
    except Exception as e:
        print(f"Could not connect to robot: {e}")
        print("Running in vision-only mode (no robot control)")
 
    # --- Camera setup ---
    cap = cv2.VideoCapture(1)
 
    with np.load('camera_params.npz') as data:
        mtx, dist_coeffs = data['mtx'], data['dist']
        dist_coeffs = dist_coeffs * 0.6
 
    h, w = None, None
    newcameramtx, roi = None, None
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        if h is None:
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist_coeffs, (w, h), 1, (w, h))
 
        undistorted = cv2.undistort(frame, mtx, dist_coeffs, None, newcameramtx)
        x, y, rw, rh = roi
        undistorted = undistorted[y:y+rh, x:x+rw]
 
        cv2.imshow("Live Webcam Feed", undistorted)
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
 
        if key == ord(' '):
            # Process frame and get paths
            # proc_frame, bot_states, grid, path1, path2, path3 = process_frame(
            #     cv2.imread("test_course.png").copy()
            # )
            proc_frame, bot_states, grid, path1, path2, path3 = process_frame(
                undistorted.copy()
            )
 
            # Display grid
            display_grid = grid.copy()
            # if path1:
            #     for (py, px) in path1:
            #         display_grid[py][px] = 2
            # if path2:
            #     for (py, px) in path2:
            #         display_grid[py][px] = 3
            if path3:
                for (py, px) in path3:
                    display_grid[py][px] = 4
 
            cv2.imshow("Processed Frame", proc_frame)
 
            plt.figure(figsize=(10, 6))
            plt.imshow(display_grid, cmap="viridis", interpolation="nearest")
            plt.title("A* Path on Occupancy Grid")
            plt.show()
 
            # --- Convert path to commands and send to robot ---
            if path1 and 4 in bot_states:
                simplified = simplify_path(path1)
                current_heading = bot_states[4]["heading"]
                commands = path_to_commands(simplified, current_heading, cell_size=5)
 
                print(f"\nBot 4 heading: {current_heading:.1f} deg")
                print(f"Simplified path: {len(simplified)} waypoints")
                print(f"Commands: {commands}\n")
 
                if robot_sock:
                    success = execute_route(robot_sock, commands)
                    if not success:
                        print("Route execution failed!")
                else:
                    print("No robot connection — commands not sent")
                    for cmd in commands:
                        print(f"  {cmd}")
 
    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    if robot_sock:
        try:
            send_command(robot_sock, 'S', 0)
        except:
            pass
        robot_sock.close()
        print("Robot disconnected")