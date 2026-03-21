import cv2
import numpy as np
import matplotlib as plt

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

grid_size = 10  # each cell = 10x10 pixels

h, w = threshold.shape
grid_h = h // grid_size
grid_w = w // grid_size

grid = np.zeros((grid_h, grid_w), dtype=int)

for cnt in contours:
    # Approximate and draw contour
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
    cv2.drawContours(output_image, [approx], 0, (0, 0, 255), 5)
    mask = np.zeros_like(threshold)

    print(cnt[0])

print(grid)


cv2.imshow("ArUco Detection Check", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
