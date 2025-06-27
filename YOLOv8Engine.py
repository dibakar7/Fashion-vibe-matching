from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load pre-trained YOLOv8 model
model = YOLO("yolov8m.pt")

# Define class names (optional: only if you want to map class IDs to names)
# You can use the COCO dataset classes or your own custom ones if the model was trained that way.
COCO_CLASSES = model.names  # built-in from the model

def FashionItemsDetector(frame_path, frame_id, conf_threshold=0.3):
    # Load the image
    image = cv2.imread(frame_path)
    if image is None:
        raise ValueError(f"Image at path '{frame_path}' could not be loaded.")

    # Convert image from BGR (OpenCV format) to RGB for correct display
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference
    results = model(rgb_image)

    # Get the first result
    result = results[0]
    detections = []

    for box in result.boxes:
        cls_id = int(box.cls)
        confidence = float(box.conf)
        if confidence < conf_threshold:
            continue

        label = COCO_CLASSES[cls_id]
        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

        detections.append({
            "label": label,
            "confidence": confidence,
            "bbox": xyxy
        })

        # Draw box and label on image
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rgb_image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return detections
