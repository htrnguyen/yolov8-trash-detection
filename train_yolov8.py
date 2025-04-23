from ultralytics import YOLO
import os

# Đảm bảo thư mục tồn tại
required_dirs = [
    './yolo_dataset/images/train',
    './yolo_dataset/images/val',
    './yolo_dataset/labels/train',
    './yolo_dataset/labels/val'
]

for dir_path in required_dirs:
    os.makedirs(dir_path, exist_ok=True)

# Tải mô hình pre-trained
model = YOLO('yolov8n.pt')

# Huấn luyện mô hình
model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    device=0,
    project='runs/train',
    name='exp'
)

print("Huấn luyện hoàn tất! Mô hình được lưu trong thư mục runs/train/exp")