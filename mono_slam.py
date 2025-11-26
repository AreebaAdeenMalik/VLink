import cv2
import numpy as np
import os


def load_camera_params(yaml_path):
    """Parses focal length and optical center."""
    if not os.path.exists(yaml_path):
        print("Config file not found. Using defaults.")
        return 718.856, 718.856, 607.1928, 185.2157

    with open(yaml_path, 'r') as f:
        try:
            # Simple parsing
            for line in f:
                if "Camera.fx:" in line: fx = float(line.split(":")[1])
                if "Camera.fy:" in line: fy = float(line.split(":")[1])
                if "Camera.cx:" in line: cx = float(line.split(":")[1])
                if "Camera.cy:" in line: cy = float(line.split(":")[1])
            return fx, fy, cx, cy
        except:
            return 500.0, 500.0, 320.0, 240.0


class VisualOdometryORB:
    def __init__(self, focal_len, pp):
        self.focal_len = focal_len
        self.pp = pp  # Principal Point (cx, cy)

        # 1. Initialize ORB
        # nfeatures=3000 ensures we find enough points even in low texture
        self.orb = cv2.ORB_create(nfeatures=3000)

        # 2. Initialize Matcher
        # Switch to standard BFMatcher (crossCheck=False allows kNN match)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # State
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None

        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.trajectory = []

    def process_frame(self, image):
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 3. Detect ORB Features & Descriptors
        kp, des = self.orb.detectAndCompute(gray, None)

        # Convert keypoints to numpy array of points for visualization/calculation
        if kp:
            kp_pts = np.float32([k.pt for k in kp])
        else:
            return None  # Totally blind

        # Initialization (First Frame)
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            # Return current position (0,0,0) and points to visualize
            return self.cur_t, kp_pts

        # 4. Match Features (Previous vs Current)
        if des is None or self.prev_des is None:
            # If we lose features, reset prev_frame to current so we can recover next frame
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return self.cur_t, kp_pts

        # Use KNN matching with k=2
        matches = self.matcher.knnMatch(self.prev_des, des, k=2)

        # Apply Lowe's Ratio Test (Robust filtering)
        good_matches = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        except ValueError:
            pass  # Handle cases where k<2

        if len(good_matches) < 10:
            # Update history even if tracking failed for this frame
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            # Not enough matches to calculate motion, but return points so user sees green dots
            return self.cur_t, kp_pts

        # Extract (x, y) coordinates from the matches
        # Note: queryIdx refers to prev_kp, trainIdx refers to current kp
        # We must perform this BEFORE updating self.prev_kp
        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in good_matches])

        # 5. Estimate Motion (Essential Matrix)
        E, mask = cv2.findEssentialMat(pts_curr, pts_prev, self.focal_len, self.pp, cv2.RANSAC, 0.999, 1.0, None)

        # Recover Rotation (R) and Translation (t)
        if E is not None:
            _, R, t, mask = cv2.recoverPose(E, pts_curr, pts_prev, focal=self.focal_len, pp=self.pp)

            # 6. Update Trajectory
            # Monocular Scale Ambiguity: We assume scale=1.0 because we have no IMU or Wheel Encoders
            absolute_scale = 1.0

            # Only update if the move is "significant" (filters jitter)
            if np.mean(np.abs(t)) > 0.005:
                self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
                self.cur_R = self.cur_R.dot(R)

            # Add to path for drawing
            # Fix DeprecationWarning by extracting scalar value first using float() or [0]
            x = int(float(self.cur_t[0])) + 400
            z = int(float(self.cur_t[2])) + 400
            self.trajectory.append((x, z))

        # 7. Update previous state AFTER all calculations
        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_des = des

        # Return the current position and ALL detected keypoints (not just matched ones)
        # This ensures the visualization always looks "active"
        return self.cur_t, kp_pts

    def draw_trajectory(self, traj_img_size=(800, 800)):
        traj_img = np.zeros((traj_img_size[0], traj_img_size[1], 3), dtype=np.uint8)

        for i in range(1, len(self.trajectory)):
            cv2.line(traj_img, self.trajectory[i - 1], self.trajectory[i], (0, 255, 0), 2)

        if self.trajectory:
            cv2.circle(traj_img, self.trajectory[-1], 5, (0, 0, 255), -1)
            cv2.putText(traj_img, f"Pos: {self.trajectory[-1]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1)

        return traj_img


def run_orb_vo_live():
    # Load params from your config file
    fx, fy, cx, cy = load_camera_params("webcam_config.yaml")
    vo = VisualOdometryORB(focal_len=fx, pp=(cx, cy))

    cap = cv2.VideoCapture(0)

    print("--- ORB Visual Odometry Started ---")
    print("Green Dots = Matched ORB Features")
    print("Move the camera slowly!")

    while True:
        ret, frame = cap.read()
        if not ret: break

        result = vo.process_frame(frame)

        display_frame = frame.copy()

        # Draw ORB Features on the camera feed
        if result is not None:
            _, current_points = result
            # Draw all features found (Green dots)
            for pt in current_points:
                cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

            # Add status text
            status_text = f"Features: {len(current_points)}"
            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Finding Features...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        traj_map = vo.draw_trajectory()

        cv2.imshow("ORB Tracking", display_frame)
        cv2.imshow("Trajectory", traj_map)

        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_orb_vo_live()