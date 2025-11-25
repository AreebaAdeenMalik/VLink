import cv2
import torch
import numpy as np


def load_midas_model(device):
    """
    Loads the MiDaS v2.1 model (CNN-based).
    """
    print("Loading MiDaS v2.1 model...")
    # Load the standard CNN-based MiDaS v2.1 model (ResNeXt-101 backbone)
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas.to(device)
    midas.eval()

    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform

    return midas, transform


def run_depth_estimation(image_path):
    # 1. Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")

    # 2. Load Model
    midas_model, midas_transform = load_midas_model(device)

    # 3. Process Image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not open {image_path}")
        return

    # Convert to RGB for processing
    img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # 4. Run Inference
    print("Running Depth Estimation...")
    input_batch = midas_transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas_model(input_batch)

        # Resize depth map to original image resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # 5. Visualization
    # Normalize depth map to 0-255 for visualization
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min)
    depth_map_vis = (depth_map_normalized * 255).astype(np.uint8)

    # Apply colormap (Magma is great for depth)
    depth_map_vis = cv2.applyColorMap(depth_map_vis, cv2.COLORMAP_MAGMA)

    # Show results
    cv2.imshow("Original Image", original_image)
    cv2.imshow("MiDaS Depth Map", depth_map_vis)

    print("\nPress any key to close the windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save result
    cv2.imwrite("results/midas_result.jpg", depth_map_vis)
    print("Saved result to midas_result.jpg")


if __name__ == "__main__":
    # Replace with your actual image path
    image_path = "results/result_yolo12.jpg"
    image_path = "resources/persons.jpg"

    import os

    if not os.path.exists(image_path):
        print(f"Creating dummy {image_path}...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.imwrite(image_path, dummy)

    run_depth_estimation(image_path)