import cv2
import os


def test_vo_single_image(image_path):
    print(f"--- Visual Odometry Initialization Test ---")
    print(f"Loading: {image_path}")

    # 1. Load Image
    if not os.path.exists(image_path):
        print(f"❌ Error: {image_path} not found.")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Error: Could not read image data.")
        return

    # 2. Convert to Grayscale (VO works on intensity, not color)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Detect Features (The "Eyes" of VO)
    # We use 'goodFeaturesToTrack' which is standard for Lucas-Kanade Optical Flow
    print("Detecting trackable features...")

    feature_params = dict(
        maxCorners=3000,
        qualityLevel=0.01,
        minDistance=9,
        blockSize=3
    )

    # This finds the strongest corners in the image
    p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

    if p0 is not None:
        print(f"✅ Success! Detected {len(p0)} features.")

        # 4. Visualize
        # Draw red circles at every detected point
        vis_frame = frame.copy()
        for i, point in enumerate(p0):
            x, y = point.ravel()
            cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        # Show result
        cv2.putText(vis_frame, f"VO Initialized: {len(p0)} Features", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Visual Odometry - Initialization", vis_frame)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save for report
        cv2.imwrite("results/vo_features.jpg", vis_frame)
        print("Saved visualization to 'vo_features.jpg'")

    else:
        print("⚠️ Warning: No features detected. Image might be too dark or plain.")


if __name__ == "__main__":
    # Generate a dummy if needed, or use your existing test_image.jpg
    img_name = "resources/persons.jpg"
    if not os.path.exists(img_name):
        import numpy as np

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw some shapes so features can be found
        cv2.rectangle(dummy, (100, 100), (300, 300), (255, 255, 255), -1)
        cv2.circle(dummy, (400, 400), 50, (255, 255, 255), -1)
        cv2.imwrite(img_name, dummy)

    test_vo_single_image(img_name)