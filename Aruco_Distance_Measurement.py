import cv2
import numpy as np
from cv2 import aruco

camera_matrix = np.array([
    [330.9382756,    0.,          307.85826339],
    [  0.,          326.26194236, 210.58270927],
    [  0.,            0.,           1.        ]
], dtype=np.float64)

dist_coeffs = np.array([
    [0.2144574, -0.49699611, -0.00350917, -0.018399, 0.27845063]
], dtype=np.float64)

# Real-world size of your ArUco marker (in meters)
MARKER_SIZE = 0.1  # 5cm marker — change to match yours

# ArUco dictionary — must match the markers you printed
aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
aruco_params = aruco.DetectorParameters()
detector     = aruco.ArucoDetector(aruco_dict, aruco_params)

# -----------------------------------------------
# Start webcam
# -----------------------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Undistort the frame using calibration data
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        for i, corner in enumerate(corners):

            # corner shape is (1, 4, 2) — 4 (x,y) pixel coords
            pts = corner[0]  # shape (4,2)

            # ----------------------------
            # Draw bounding box
            # ----------------------------
            pts_int = pts.astype(int)
            cv2.polylines(frame, [pts_int], isClosed=True,
                          color=(0, 255, 0), thickness=2)

            # Label the marker ID at the top-left corner
            top_left = tuple(pts_int[0])
            cv2.putText(frame, f"ID: {ids[i][0]}",
                        (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

            # ----------------------------
            # Estimate pose (rvec, tvec)
            # ----------------------------
            # Define the 3D coordinates of the marker corners
            # in the marker's own coordinate frame (meters)
            half = MARKER_SIZE / 2
            obj_points = np.array([
                [-half,  half, 0],
                [ half,  half, 0],
                [ half, -half, 0],
                [-half, -half, 0]
            ], dtype=np.float32)

            img_points = pts.astype(np.float32)

            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_points,
                camera_matrix, dist_coeffs
            )

            if success:
                # tvec[2] is the Z component = depth = distance
                distance = float(tvec[2][0])
                offset_x = float(tvec[0][0])  # negative = left, positive = right

                # Convert rvec to rotation matrix, then extract yaw (Z rotation)
                rot_matrix, _ = cv2.Rodrigues(rvec)

                # Decompose rotation matrix into Euler angles (radians)
                # sy = cosine of pitch, used to detect gimbal lock
                sy = np.sqrt(rot_matrix[0, 0] ** 2 + rot_matrix[1, 0] ** 2)
                singular = sy < 1e-6  # true if gimbal lock

                if not singular:
                    yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
                    pitch = np.arctan2(-rot_matrix[2, 0], sy)
                    roll = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
                else:
                    yaw = np.arctan2(-rot_matrix[1, 2], rot_matrix[1, 1])
                    pitch = np.arctan2(-rot_matrix[2, 0], sy)
                    roll = 0.0

                # Convert radians to degrees
                yaw_deg = np.degrees(yaw)
                pitch_deg = np.degrees(pitch)
                roll_deg = np.degrees(roll)

                # Determine left/right label
                if offset_x < -0.01:
                    h_label = f"Left  {abs(offset_x):.3f}m"
                elif offset_x > 0.01:
                    h_label = f"Right {abs(offset_x):.3f}m"
                else:
                    h_label = "Center"

                # Display all values on frame
                bottom_left = tuple(pts_int[3])
                cv2.putText(frame, f"H-Off: {h_label}",
                            (bottom_left[0], bottom_left[1] + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Yaw:   {yaw_deg:.1f}deg",
                            (bottom_left[0], bottom_left[1] + 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(frame, f"Pitch: {pitch_deg:.1f}deg",
                            (bottom_left[0], bottom_left[1] + 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(frame, f"Roll:  {roll_deg:.1f}deg",
                            (bottom_left[0], bottom_left[1] + 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # Draw distance text below the marker
                bottom_left = tuple(pts_int[3])
                cv2.putText(frame, f"Dist: {distance:.2f}m",
                            (bottom_left[0], bottom_left[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

                # Optionally draw the pose axes
                cv2.drawFrameAxes(frame, camera_matrix,
                                  dist_coeffs, rvec, tvec,
                                  MARKER_SIZE * 0.5)

    cv2.imshow("ArUco Distance Measurement", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
