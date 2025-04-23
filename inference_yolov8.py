import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Tải mô hình đã huấn luyện
model = YOLO('./yolov8_trained.pt')

# Đường dẫn đến thư mục chứa ảnh kiểm tra
test_img_dir = './yolo_dataset/images/val'
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# Lấy danh sách ảnh kiểm tra (giới hạn 5 ảnh để hiển thị nhanh)
test_imgs = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]

# Suy luận trên từng ảnh
for img_path in test_imgs:
    # Đọc ảnh
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Suy luận với YOLOv8
    results = model.predict(img_path, conf=0.5, iou=0.6)

    # Vẽ hộp giới hạn và nhãn
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Hộp giới hạn
        labels = result.boxes.cls.cpu().numpy()  # Nhãn
        scores = result.boxes.conf.cpu().numpy()  # Độ tin cậy

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = map(int, box)
            class_name = result.names[int(label)]

            # Vẽ hộp giới hạn
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_rgb, f"{class_name} ({score:.2f})", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị ảnh
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title(f'Ảnh: {os.path.basename(img_path)}')
    plt.axis('off')
    plt.show()

    # Lưu ảnh kết quả
    save_path = os.path.join(output_dir, os.path.basename(img_path).replace('.jpg', '_detected.jpg'))
    cv2.imwrite(save_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    print(f"Đã lưu ảnh tại: {save_path}")