from ultralytics import YOLO
import os

# Get absolute path to project root
ROOT = os.path.dirname(os.path.abspath(__file__))

# Define directories with absolute paths
required_dirs = [
    os.path.join(ROOT, 'yolo_dataset/images/train'),
    os.path.join(ROOT, 'yolo_dataset/images/val'),
    os.path.join(ROOT, 'yolo_dataset/labels/train'),
    os.path.join(ROOT, 'yolo_dataset/labels/val')
]

# Create directories
for dir_path in required_dirs:
    os.makedirs(dir_path, exist_ok=True)

# Load pre-trained model
model = YOLO('yolov8n.pt')

# Train model with absolute paths
model.train(
    data=os.path.join(ROOT, 'data.yaml'),
    epochs=50,
    imgsz=640,
    device=0,
    project=os.path.join(ROOT, 'runs/train'),
    name='exp'
)

print("Training completed! Model saved in runs/train/exp")