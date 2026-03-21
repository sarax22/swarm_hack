import cv2 as cv
import numpy as np

image = cv.imread('test_course.png')
arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
arucoParams = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)
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
    cv.aruco.drawDetectedMarkers(output_image, corners, ids)

    for i, (corner, marker_id) in enumerate(zip(corners, ids)):
        pts = corner.reshape((4, 2))
        cx = int((pts[0][0] + pts[2][0]) / 2)
        cy = int((pts[0][1] + pts[2][1]) / 2)

        # Draw center point
        cv.circle(output_image, (cx, cy), 5, (0, 255, 0), -1)

        # Label center coordinates
        # cv.putText(
        #     output_image,
        #     f"ID{marker_id[0]} ({cx},{cy})",
        #     (cx + 8, cy - 8),
        #     cv.FONT_HERSHEY_SIMPLEX,
        #     0.5, (0, 0, 255), 2
        # )


blur = cv.GaussianBlur(output_image, (5, 5), 1.4)
edges = cv.Canny(blur, threshold1=100, threshold2=200)
cv.imshow("ArUco Detection Check", edges)
cv.waitKey(0)
cv.destroyAllWindows()