import cv2
import numpy as np
import yaml
import os


def load_camera_params(yaml_path):
    """Parses the focal length and optical center from the yaml file."""
    if not os.path.exists(yaml_path):
        print("Config file not found. Using defaults.")
        return 718.856, 718.856, 607.1928, 185.2157  # KITTI defaults as fallback

    with open(yaml_path, 'r') as f:
        # Simple line parsing to avoid dependency issues if PyYAML isn't perfect
        data = f.read()

    # Extract values manually or via yaml
    # Assuming standard format from our previous script
    try:
        params = yaml.safe_load(data)
        fx = float(params.get('Camera.fx', 500.0))
        fy = float(params.get('Camera.fy', 500.0))
        cx = float(params.get('Camera.cx', 320.0))
        cy = float(params.get('Camera.cy', 240.0))
        return fx, fy, cx, cy
    except:
        return 500.0, 500.0, 320.0, 240.0


class VisualOdometry:
    def __init__(self, focal_len, pp):
        self.focal_len = focal_len
        self.pp = pp  # Principal Point (cx, cy)
        self.orb = cv2.ORB_create(3000)
        self.lk_params = dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        # State
        self.prev_image = None
        self.prev_keypoints = None
        self.cur_R = np.eye(3)  # Current Rotation
        self.cur_t = np.zeros((3, 1))  # Current Translation
        self.trajectory = []  # Path history

    def process_frame(self, image):
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Initialization (First Frame)
        if self.prev_image is None:
            self.prev_image = gray
            # Detect Features (Good Features to Track is more stable for VO than ORB)
            self.prev_keypoints = cv2.goodFeaturesToTrack(gray, maxCorners=3000, qualityLevel=0.01, minDistance=10)
            return None

        # 1. Track Features (Optical Flow)
        # We track points from Previous Frame -> Current Frame
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_image, gray, self.prev_keypoints, None, **self.lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.prev_keypoints[st == 1]

        # 2. Estimate Motion (Essential Matrix)
        # This calculates the geometric transformation between frames
        if len(good_new) < 10:
            # Lost tracking
            self.prev_image = gray
            self.prev_keypoints = cv2.goodFeaturesToTrack(gray, maxCorners=3000, qualityLevel=0.01, minDistance=10)
            return None

        E, mask = cv2.findEssentialMat(good_new, good_old, self.focal_len, self.pp, cv2.RANSAC, 0.999, 1.0, None)

        # Recover Rotation (R) and Translation (t) from Essential Matrix
        _, R, t, mask = cv2.recoverPose(E, good_new, good_old, focal=self.focal_len, pp=self.pp)

        # 3. Update Trajectory
        # Absolute scale is unknown in Monocular VO, so we assume unit scale or use speed heuristic
        absolute_scale = 1.0

        # Update global position
        # t is the relative motion. We add it to our global 'cur_t'
        if (absolute_scale > 0.1):
            self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
            self.cur_R = self.cur_R.dot(R)

        # Save path for drawing
        self.trajectory.append((int(self.cur_t[0]) + 400, int(self.cur_t[2]) + 400))

        # Update previous state for next loop
        self.prev_image = gray
        self.prev_keypoints = good_new.reshape(-1, 1, 2)

        return self.cur_t

    def draw_trajectory(self, traj_img_size=(800, 800)):
        # Create a blank map
        traj_img = np.zeros((traj_img_size[0], traj_img_size[1], 3), dtype=np.uint8)

        # Draw path
        for i in range(1, len(self.trajectory)):
            cv2.line(traj_img, self.trajectory[i - 1], self.trajectory[i], (0, 255, 0), 2)

        # Draw current pos
        if self.trajectory:
            cv2.circle(traj_img, self.trajectory[-1], 5, (0, 0, 255), -1)

        return traj_img


def run_vo_live():
    # Load params
    fx, fy, cx, cy = load_camera_params("webcam_config.yaml")
    print(f"Camera: fx={fx}, cx={cx}, cy={cy}")

    vo = VisualOdometry(focal_len=fx, pp=(cx, cy))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening camera")
        return

    print("Starting Visual Odometry...")
    print("👉 Move the camera smoothly to see the path.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run VO
        translation = vo.process_frame(frame)

        # Visualization
        # 1. Draw tracked features on the camera feed
        display_frame = frame.copy()
        if vo.prev_keypoints is not None:
            for point in vo.prev_keypoints:
                x, y = point.ravel()
                cv2.circle(display_frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        # 2. Get the Trajectory Map
        traj_map = vo.draw_trajectory()

        # Show windows
        cv2.imshow("Camera Feed (Tracking)", display_frame)
        cv2.imshow("Trajectory Map (Top-Down)", traj_map)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_vo_live()