from ultralytics import YOLO
import os

# Đường dẫn đến tệp data.yaml
data_yaml_path = './data.yaml'

# Tải mô hình pre-trained YOLOv8 (phiên bản nano để tiết kiệm tài nguyên trên GPU P100)
model = YOLO('yolov8n.pt')

# Huấn luyện mô hình
# - epochs: Số vòng huấn luyện (dùng 50 để đảm bảo hội tụ tốt)
# - imgsz: Kích thước ảnh (dùng 640 để tiết kiệm tài nguyên)
# - device: 0 (GPU P100)
model.train(data=data_yaml_path, epochs=50, imgsz=640, device=0)

# Lưu mô hình đã huấn luyện
model.save('./yolov8_trained.pt')
print("Huấn luyện YOLOv8 hoàn tất! Mô hình được lưu tại /kaggle/working/yolov8_trained.pt")