import os
import json
import labelme2coco

# Đường dẫn đến thư mục chứa dữ liệu
dataset_dir = './dataset'
output_json_path = './dataset/coco_output'

def check_annotations(dataset_dir):
    problems = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(dataset_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for shape in data['shapes']:
                    if len(shape['points']) < 4:
                        problems.append((filename, len(shape['points'])))
    
    return problems

# Kiểm tra trước khi chuyển đổi
print("Đang kiểm tra annotations...")
problems = check_annotations(dataset_dir)
if problems:
    print("Tìm thấy các file có vấn đề:")
    for filename, point_count in problems:
        print(f"File {filename}: có polygon chỉ có {point_count} điểm")
    exit(1)

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

# Chuyển đổi từ LabelMe sang COCO
labelme2coco.convert(dataset_dir, output_json_path)
print("Chuyển đổi từ LabelMe sang COCO hoàn tất!")