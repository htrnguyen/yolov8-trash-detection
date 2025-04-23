import json
import os
import random
import shutil
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split

# Đường dẫn đến tệp COCO và thư mục đầu ra
coco_json_path = './dataset/coco_output/dataset.json'
images_dir = './dataset/images'
output_base_dir = './yolo_dataset'
os.makedirs(output_base_dir, exist_ok=True)

# Tạo thư mục cho train và val
train_img_dir = os.path.join(output_base_dir, 'images/train')
val_img_dir = os.path.join(output_base_dir, 'images/val')
train_label_dir = os.path.join(output_base_dir, 'labels/train')
val_label_dir = os.path.join(output_base_dir, 'labels/val')
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Tải tệp COCO
coco = COCO(coco_json_path)

# Lấy danh sách category
categories = coco.loadCats(coco.getCatIds())
category_map = {cat['id']: idx for idx, cat in enumerate(categories)}

# Lấy danh sách tất cả ảnh
img_ids = coco.getImgIds()
imgs = coco.loadImgs(img_ids)

# Chia tập train và val (80% train, 20% val)
train_imgs, val_imgs = train_test_split(imgs, test_size=0.2, random_state=42)

# Hàm chuyển đổi và lưu ảnh/nhãn
def convert_and_save(img_list, img_output_dir, label_output_dir):
    for img in img_list:
        img_id = img['id']
        img_width, img_height = img['width'], img['height']
        
        # Xử lý tên file đúng cách
        img_filename = os.path.basename(img['file_name'])
        img_path = os.path.join(images_dir, img_filename)
        output_img_path = os.path.join(img_output_dir, img_filename)

        # Kiểm tra và copy file
        if os.path.exists(img_path):
            try:
                shutil.copy2(img_path, output_img_path)
                print(f"Đã copy {img_filename} thành công")
                
                # Tạo nhãn YOLO
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                txt_path = os.path.join(
                    label_output_dir, 
                    os.path.splitext(img_filename)[0] + '.txt'
                )
                
                with open(txt_path, 'w') as f:
                    for ann in anns:
                        bbox = ann['bbox']
                        x_center = (bbox[0] + bbox[2] / 2) / img_width
                        y_center = (bbox[1] + bbox[3] / 2) / img_height
                        width = bbox[2] / img_width
                        height = bbox[3] / img_height
                        class_id = category_map[ann['category_id']]
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                print(f"Đã tạo nhãn cho {img_filename}")
            except Exception as e:
                print(f"Lỗi khi xử lý {img_filename}: {str(e)}")
        else:
            print(f"Không tìm thấy file: {img_path}")

# Chuyển đổi và lưu cho tập train
convert_and_save(train_imgs, train_img_dir, train_label_dir)
print("Đã chuyển đổi và lưu tập train!")

# Chuyển đổi và lưu cho tập val
convert_and_save(val_imgs, val_img_dir, val_label_dir)
print("Đã chuyển đổi và lưu tập val!")