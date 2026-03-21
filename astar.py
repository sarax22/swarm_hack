import heapq
import numpy as np

def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    rows, cols = grid.shape

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}

    # 4-direction movement (change to 8-dir if needed)
    neighbors = [(0,1),(1,0),(0,-1),(-1,0)]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        cy, cx = current

        for dy, dx in neighbors:
            ny, nx = cy + dy, cx + dx

            # skip invalid positions
            if not (0 <= ny < rows and 0 <= nx < cols):
                continue

            # skip obstacles
            if grid[ny][nx] == 1:
                continue

            tentative_g = gscore[current] + 1

            if (ny, nx) not in gscore or tentative_g < gscore[(ny, nx)]:
                came_from[(ny, nx)] = current
                gscore[(ny, nx)] = tentative_g
                fscore[(ny, nx)] = tentative_g + heuristic((ny, nx), goal)
                heapq.heappush(open_set, (fscore[(ny, nx)], (ny, nx)))

    return None  # no path found