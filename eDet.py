import torch
import cv2
import numpy as np
from effdet import create_model
import os
from torchvision import transforms


# Removed matplotlib to avoid PyCharm backend errors

# 1. Load the Model
def load_model(model_name='tf_efficientdet_d0', device='cuda'):
    """
    Loads a pretrained EfficientDet model.
    """
    print(f"Loading {model_name}...")
    # create_model loads the structure and downloads pretrained weights
    # bench_task='predict' sets it up for inference (no training heads)
    model = create_model(model_name, bench_task='predict', pretrained=True, num_classes=90)

    model.to(device)
    model.eval()  # Set to evaluation mode
    return model


# 2. Preprocess Image
def preprocess_image(image, input_size=(512, 512), device='cuda'):
    """
    Resizes and normalizes the image for EfficientDet.
    """
    # EfficientDet expects a specific normalization mean/std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Apply transforms
    input_tensor = transform(image)

    # Add batch dimension (1, C, H, W)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    return input_tensor


# 3. Run Inference
def detect_objects(model, image_path, confidence_threshold=0.5, device='cuda'):
    """
    Runs the full detection pipeline on a single image.
    """
    # Read image using OpenCV
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Could not find image at {image_path}")

    # Convert BGR (OpenCV) to RGB
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Preprocess
    # D0 uses 512x512 by default. D1=640, D2=768, etc.
    input_tensor = preprocess_image(rgb_image, input_size=(512, 512), device=device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)

    # Output format: [Batch, Num_Detections, 6]
    # The 6 values are: [x_min, y_min, x_max, y_max, score, class_id]
    detections = output[0].cpu().numpy()

    # 4. Filter & Display Results
    # COCO Classes (Simple list for demo purposes)
    # Note: Full list is 90 classes. This is a truncated example.
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]

    print(f"\n--- Detections (Confidence > {confidence_threshold}) ---")

    # Scale boxes back to original image size
    h_scale = original_image.shape[0] / 512
    w_scale = original_image.shape[1] / 512

    for i in range(detections.shape[0]):
        # Unpack the row
        x_min, y_min, x_max, y_max, score, class_id = detections[i]

        if score > confidence_threshold:
            # Scale coords
            x_min *= w_scale
            x_max *= w_scale
            y_min *= h_scale
            y_max *= h_scale

            # Draw box
            start_point = (int(x_min), int(y_min))
            end_point = (int(x_max), int(y_max))
            color = (0, 255, 0)  # Green
            thickness = 2

            cv2.rectangle(original_image, start_point, end_point, color, thickness)

            # Label
            label_idx = int(class_id) - 1  # effdet classes are 1-indexed often
            label_text = f"{COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else 'Unknown'}: {score:.2f}"

            cv2.putText(original_image, label_text, (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            print(f"Found {label_text} at [{x_min:.0f}, {y_min:.0f}, {x_max:.0f}, {y_max:.0f}]")

    # Show final image using OpenCV (Standard for Computer Vision)
    # This avoids the 'FigureCanvasInterAgg' error in PyCharm
    cv2.imshow("EfficientDet Predictions", original_image)
    print("\nPress any key to close the image window...")
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()

    # Save result
    cv2.imwrite("result.jpg", original_image)
    print("\nResult saved to result.jpg")


if __name__ == "__main__":
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")

    # Initialize
    model = load_model(model_name='tf_efficientdet_d0', device=device)

    detect_objects(model, "resources/RXh4tdqXZpxUfoq6mNTbGo.jpg", confidence_threshold=0.3, device=device)