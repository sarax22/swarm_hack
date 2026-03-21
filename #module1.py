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

if ids is not None:
    cv2.aruco.drawDetectedMarkers(output_image, corners, ids)

    for i, (corner, marker_id) in enumerate(zip(corners, ids)):
        pts = corner.reshape((4, 2))
        cx = int((pts[0][0] + pts[2][0]) / 2)
        cy = int((pts[0][1] + pts[2][1]) / 2)

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

font = cv2.FONT_HERSHEY_COMPLEX
img = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


HEIGHT, WIDTH = threshold.shape  # use your actual image size
CHUNK = 10

CHUNK_H = HEIGHT // CHUNK
CHUNK_W = WIDTH // CHUNK

grid = np.zeros((CHUNK_H, CHUNK_W), dtype=int)


for cnt in contours:
    # Approximate and draw contour
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
    cv2.drawContours(output_image, [approx], 0, (0, 0, 255), 5)
    
    mask = np.zeros_like(threshold)
    cv2.drawContours(mask, contours, -1, 255, 1)   # thickness=1 → single-pixel contour lines

    # ---- 2. For each pixel in mask that is white (part of a line), mark its chunk ----
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

start = (13,65)
goal = (77,55)


path = astar(grid, start, goal)


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
if path != None:
    for (y, x) in path:
        display_grid[y][x] = 2   # mark path cells
else:
    print("aaaaaaaaaaaaaaaaa")

plt.figure(figsize=(10,6))
plt.imshow(display_grid, cmap="viridis", interpolation="nearest")
plt.title("A* Path on Occupancy Grid")
plt.show()
