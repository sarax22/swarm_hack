import cv2
import numpy as np
import os
import time

def setup_camera():
    cap = cv2.VideoCapture(2)

    with np.load('camera_params.npz') as data:
        mtx, dist = data['mtx'], data['dist']
        dist = dist * 0.6

    os.makedirs("screenshots", exist_ok=True)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]

        # cv2.imshow('Original', frame)
        # cv2.imshow('Undistorted', undistorted)

        return ret, frame

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = f"screenshots/screenshot_{count:03d}_{ts}.png"
            cv2.imwrite(path, undistorted)
            count += 1
            print(f"Saved: {path}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

setup_camera()
