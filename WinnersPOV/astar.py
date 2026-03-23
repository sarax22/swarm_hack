"""
A* pathfinding with 8-directional movement and line-of-sight smoothing.
Produces shortest paths with precise angles instead of 90-degree grid paths.
"""
 
import heapq
import math
import numpy as np
 
 
# 8-directional movement: (dy, dx, cost)
# Cardinal = 1.0, Diagonal = sqrt(2)
DIRECTIONS_8 = [
    (-1,  0, 1.0),    # N
    ( 1,  0, 1.0),    # S
    ( 0, -1, 1.0),    # W
    ( 0,  1, 1.0),    # E
    (-1, -1, 1.414),  # NW
    (-1,  1, 1.414),  # NE
    ( 1, -1, 1.414),  # SW
    ( 1,  1, 1.414),  # SE
]
 
 
def heuristic(a, b):
    """Octile distance — admissible heuristic for 8-directional grid."""
    dy = abs(a[0] - b[0])
    dx = abs(a[1] - b[1])
    return max(dy, dx) + (1.414 - 1.0) * min(dy, dx)
 
 
def astar(grid, start, goal):
    """
    A* on an occupancy grid with 8-directional movement.
    
    Args:
        grid: 2D numpy array or list-of-lists. 0 = free, 1 = obstacle.
        start: (row, col) tuple
        goal:  (row, col) tuple
    
    Returns:
        List of (row, col) from start to goal, or None if no path.
    """
    rows = len(grid)
    cols = len(grid[0])
 
    # Bounds check
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return None
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return None
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        return None
 
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
 
    while open_set:
        _, current = heapq.heappop(open_set)
 
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
 
        for dy, dx, cost in DIRECTIONS_8:
            ny, nx = current[0] + dy, current[1] + dx
 
            if not (0 <= ny < rows and 0 <= nx < cols):
                continue
            if grid[ny][nx] == 1:
                continue
 
            # For diagonal moves, check that both adjacent cardinal cells are free
            # This prevents cutting corners around obstacles
            if dy != 0 and dx != 0:
                if grid[current[0] + dy][current[1]] == 1 or grid[current[0]][current[1] + dx] == 1:
                    continue
 
            neighbor = (ny, nx)
            tentative_g = g_score[current] + cost
 
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))
 
    return None  # No path found
 
 
def line_of_sight(grid, p1, p2):
    """
    Bresenham line check — returns True if there's a clear line
    between p1 and p2 on the grid (no obstacles).
    p1, p2: (row, col) tuples
    """
    r0, c0 = p1
    r1, c1 = p2
    
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    
    err = dr - dc
    r, c = r0, c0
    
    rows = len(grid)
    cols = len(grid[0])
    
    while True:
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        if grid[r][c] == 1:
            return False
        if r == r1 and c == c1:
            return True
        
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
 
 
def smooth_path(grid, path):
    """
    Line-of-sight path smoothing. Removes unnecessary waypoints
    so the robot takes straight-line shortcuts instead of following
    the grid staircase.
    
    Greedy approach: from current point, find the farthest visible
    point and jump there.
    """
    if not path or len(path) <= 2:
        return path
 
    smoothed = [path[0]]
    current_idx = 0
 
    while current_idx < len(path) - 1:
        # Find the farthest point we can see from current
        farthest = current_idx + 1
        for check_idx in range(len(path) - 1, current_idx, -1):
            if line_of_sight(grid, path[current_idx], path[check_idx]):
                farthest = check_idx
                break
        
        smoothed.append(path[farthest])
        current_idx = farthest
 
    return smoothed
