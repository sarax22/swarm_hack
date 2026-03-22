import heapq
import numpy as np

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal, turn_penalty=0.01):
    rows, cols = grid.shape

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    came_dir = {}
    gscore = {start: 0}

    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        cy, cx = current
        prev_dir = came_dir.get(current, None)

        for dy, dx in neighbors:
            ny, nx = cy + dy, cx + dx

            if not (0 <= ny < rows and 0 <= nx < cols):
                continue
            if grid[ny][nx] == 1:
                continue

            direction = (dy, dx)
            is_turn = prev_dir is not None and direction != prev_dir
            tentative_g = gscore[current] + 1 + (turn_penalty if is_turn else 0)

            if (ny, nx) not in gscore or tentative_g < gscore[(ny, nx)]:
                came_from[(ny, nx)] = current
                came_dir[(ny, nx)] = direction
                gscore[(ny, nx)] = tentative_g
                f = tentative_g + heuristic((ny, nx), goal)
                heapq.heappush(open_set, (f, (ny, nx)))

    return None