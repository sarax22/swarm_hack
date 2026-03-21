import cv2
import numpy as np
import matplotlib.pyplot as plt
from astar import astar

image = cv2.imread('test_course.png')
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
(corners, ids, rejected) = detector.detectMarkers(image)

if ids is not None:
    print(f"Successfully detected {len(ids)} markers.\n")

    for i, (corner, marker_id) in enumerate(zip(corners, ids)):
        # corners[i] shape: (1, 4, 2) → squeeze to (4, 2)
        pts = corner.reshape((4, 2))

        top_left     = pts[0]
        top_right    = pts[1]
        bottom_right = pts[2]
        bottom_left  = pts[3]

        # Center of the marker
        cx = int((top_left[0] + bottom_right[0]) / 2)
        cy = int((top_left[1] + bottom_right[1]) / 2)

        print(f"Marker ID: {marker_id[0]}")
        print(f"  Top-Left:     ({int(top_left[0])},     {int(top_left[1])})")
        print(f"  Top-Right:    ({int(top_right[0])},    {int(top_right[1])})")
        print(f"  Bottom-Right: ({int(bottom_right[0])}, {int(bottom_right[1])})")
        print(f"  Bottom-Left:  ({int(bottom_left[0])},  {int(bottom_left[1])})")
        print(f"  Center:       ({cx}, {cy})")
        print()
else:
    print("No markers detected.")


output_image = image.copy()

bots = {}
if ids is not None:
    cv2.aruco.drawDetectedMarkers(output_image, corners, ids)

    for i, (corner, marker_id) in enumerate(zip(corners, ids)):
        pts = corner.reshape((4, 2))
        cx = int((pts[0][0] + pts[2][0]) / 2)
        cy = int((pts[0][1] + pts[2][1]) / 2)

        bots[marker_id[0]] = (cx,cy)


        # Draw center point
        cv2.circle(output_image, (cx, cy), 5, (0, 255, 0), -1)

        # Label center coordinates
        # cv2.putText(
        #     output_image,
        #     f"ID{marker_id[0]} ({cx},{cy})",
        #     (cx + 8, cy - 8),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5, (0, 0, 255), 2
        # )

marker_centers = list(bots.values())

# --- Detect red regions ---
hsv = cv2.cvtColor(output_image, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 150, 80])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 150, 80])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = mask1 | mask2

# ---- EDGE detection instead of grayscale threshold ----
edges = cv2.Canny(red_mask, 50, 150)

cv2.imshow("EDGE MASK", edges)
cv2.waitKey(0)

# ---- Contours from edges ----
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


HEIGHT, WIDTH = edges.shape  # use your actual image size
CHUNK = 5

CHUNK_H = HEIGHT // CHUNK
CHUNK_W = WIDTH // CHUNK

grid = np.zeros((CHUNK_H, CHUNK_W), dtype=int)

max_y, max_x, _ = output_image.shape
mask = np.zeros_like(edges)


cv2.imshow("RED MASK", red_mask)
cv2.waitKey(0)


# Draw only red contours
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

    # draw on the output-visible image
    cv2.drawContours(output_image, [approx], -1, (0, 0, 255), 5)

    # draw ONLY this contour on mask
    cv2.drawContours(mask, [cnt], -1, 255, 1)

# Exclude ArUco areas (run ONCE)
EXCLUDE_RADIUS = 60
for (mx, my) in marker_centers:
    cv2.circle(mask, (mx, my), EXCLUDE_RADIUS, 0, -1)

# Build occupancy grid (run ONCE)
ys, xs = np.where(mask == 255)

for x, y in zip(xs, ys):
    cx = min(x // CHUNK, CHUNK_W - 1)
    cy = min(y // CHUNK, CHUNK_H - 1)
    grid[cy, cx] = 1



    
    # if 0 <= cnt[0][0] < WIDTH and 0 <= cnt[0][1] < HEIGHT:
    #         cx = cnt[0][0] // CHUNK
    #         cy = cnt[0][1] // CHUNK
    #         grid[cy, cx] = 1


bots = {}
for i, (corner, marker_id) in enumerate(zip(corners, ids)):
        # corners[i] shape: (1, 4, 2) → squeeze to (4, 2)
        pts = corner.reshape((4, 2))



        print(marker_id)
        cx = int((pts[0][0] + pts[2][0]) / 2)
        cy = int((pts[0][1] + pts[2][1]) / 2)

        print(cx,cy)

        bots[marker_id[0]] = (cx,cy)

print(bots)

inflate_radius = 3 
grid_img = (grid.astype(np.uint8)) * 255
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflate_radius*2+1, inflate_radius*2+1))
inflated = cv2.dilate(grid_img, kernel)
inflated_grid = (inflated > 0).astype(int)

start1 = (bots[4][1]//CHUNK,bots[4][0]//CHUNK)
start2 = (bots[2][1]//CHUNK,bots[2][0]//CHUNK)
start3 = (bots[3][1]//CHUNK,bots[3][0]//CHUNK)
goal = (bots[7][1]//CHUNK,bots[7][0]//CHUNK)


path1 = astar(inflated_grid, start1, goal)
path2 = astar(inflated_grid, start2, goal)
path3 = astar(inflated_grid, start3, goal)


cv2.imshow("ArUco Detection Check", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


plt.figure(figsize=(10, 6))
plt.imshow(grid, cmap="Greys", interpolation="nearest")
plt.title("Contour-Based 10×10 Occupancy Grid")
plt.xlabel("Chunk X")
plt.ylabel("Chunk Y")
plt.grid(color='lightgray', linestyle='--', linewidth=0.3)
plt.show()


display_grid = grid.copy()
if path1 != None:
    for (y, x) in path1:
        display_grid[y][x] = 2   # mark path cells

if path2 != None:
    for (y, x) in path2:
        display_grid[y][x] = 3   # mark path cells

if path3 != None:
    for (y, x) in path3:
        display_grid[y][x] = 4   # mark path cells

print("1",path1)
print("2",path2)
print("3",path3)
print(goal)
plt.figure(figsize=(10,6))
plt.imshow(display_grid, cmap="viridis", interpolation="nearest")
plt.title("A* Path on Occupancy Grid")
plt.show()
