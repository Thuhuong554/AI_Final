import os
import cv2
import mediapipe as mp
import csv
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Chỉ hiện lỗi (ERROR)

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# Hàm tính toán landmark list
def calc_landmark_list(image, hand_landmarks):
    image_height, image_width = image.shape[:2]
    landmark_list = []
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)
        landmark_list.append([x, y])
    return landmark_list

# Hàm xử lý pre-process landmark
def pre_process_landmark(landmark_list):
    # Chuyển về gốc (0, 0)
    base_x, base_y = landmark_list[0]
    relative_landmarks = []
    for x, y in landmark_list:
        relative_landmarks.append([x - base_x, y - base_y])
    
    # Chuẩn hóa dữ liệu
    flattened = np.array(relative_landmarks).flatten()
    max_value = max(abs(flattened))
    normalized_landmarks = (flattened / max_value).tolist()
    return normalized_landmarks

# Kiểm tra xem file có phải ảnh hay không
def is_image(file_path):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    return any(file_path.lower().endswith(ext) for ext in valid_extensions)

# Duyệt qua dataset và lưu vào CSV
def process_dataset(input_dir, output_csv):
    with open(output_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['label'] + [f'feature_{i}' for i in range(42)])  # 21 điểm x 2 tọa độ

        # Duyệt qua các thư mục lớp trong dataset
        for label_folder in os.listdir(input_dir):
            label_path = os.path.join(input_dir, label_folder)
            if not os.path.isdir(label_path):
                continue
            print(f'Processing folder: {label_folder}')  # In thư mục lớp

            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                
                # Kiểm tra xem file có phải ảnh không
                if not is_image(image_path):
                    print(f'Skipping non-image file: {image_path}')  # In ra thông báo bỏ qua file không phải ảnh
                    continue

                image = cv2.imread(image_path)
                if image is None:
                    print(f'Failed to load image: {image_path}')  # Nếu ảnh không tải được
                    continue

                print(f'Processing image: {image_file}')  # In tên ảnh

                # Xử lý ảnh với Mediapipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                print(f'Results: {results}')  # In tên ảnh
                print(f'results.multi_hand_landmarks: {results.multi_hand_landmarks}')  # In tên ảnh


                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmark_list = calc_landmark_list(image, hand_landmarks)
                        print(f'Landmarks: {landmark_list}')  # In ra các landmark
                        processed_landmarks = pre_process_landmark(landmark_list)
                        print(f'Processed landmarks: {processed_landmarks}')  # In ra các landmark đã xử lý
                        writer.writerow([label_folder] + processed_landmarks)

# Chạy chương trình
# Dataset input path
input_directory = r'C:\Users\admin\.cache\kagglehub\datasets\kirlelea\spanish-sign-language-alphabet-static\versions\1\fondo_blanco'

# Output CSV path
output_csv_file = 'C:/Users/admin/OneDrive/Máy tính/hand-gesture-recognition-mediapipe-main/hand-gesture-recognition-mediapipe-main/model/keypoint_classifier_v1/keypoint_classifier_v1.csv'

process_dataset(input_directory, output_csv_file)

hands.close()
