from ultralytics import YOLO
import cv2
import torch


def run_yolo_inference(image_path):
    # 1. Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")

    # 2. Load the Model
    # Switching to YOLO12 (State-of-the-Art as of 2025)
    # 'Yolov12n.pt' = Nano (Fastest)
    # 'Yolov12s.pt' = Small (Balanced)
    # 'Yolov12m.pt' = Medium
    print("Loading YOLO12 model...")
    model = YOLO("yolo12l.pt")

    # 3. Run Inference
    results = model(image_path, device=device)

    # 4. Visualize
    annotated_frame = results[0].plot()

    # 5. Display
    cv2.imshow("Original Image", cv2.imread(image_path))
    cv2.imshow("YOLO12 Detections", annotated_frame)

    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save result
    cv2.imwrite("results/result_yolo12.jpg", annotated_frame)
    print("Saved result to result_yolo12.jpg")


if __name__ == "__main__":
    # Replace with your actual image path
    image_path = "resources/persons.jpg"

    run_yolo_inference(image_path)
