#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
from model import KeyPointClassifierV1

# Define colors for modern aesthetics
COLORS = {
    "background": (0, 0, 0),
    "line_primary": (255, 255, 255),
    "line_secondary": (50, 200, 255),
    "circle": (255, 255, 255),
    "text": (0, 255, 0),
    "highlight": (152, 251, 152),
}

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifierV1()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier_v1/keypoint_classifier_v1_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0
    #  ############################################################
    recognized_text = []

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == "Not applicable":  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])
                    recognized_text = update_recognized_text(recognized_text, hand_sign_id, keypoint_classifier_labels)

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
                # Hiển thị kết quả
                debug_image = draw_recognized_text(debug_image, recognized_text)

        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier_v1/keypoint_classifier_v1.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return

import cv2 as cv

def draw_landmarks(image, landmark_point):
    def draw_line(image, point1, point2, color, thickness):
        """Vẽ một đường với màu sắc tùy chỉnh."""
        cv.line(image, tuple(point1), tuple(point2), color, thickness)

    def draw_circle(image, point, radius, color):
        """Vẽ một vòng tròn với màu sắc tùy chỉnh."""
        cv.circle(image, tuple(point), radius, color, -1)
        cv.circle(image, tuple(point), radius, (0, 0, 0), 1)  # Viền đen để nổi bật

    # Màu sắc chuẩn Mediapipe
    mp_colors = {
        "thumb": (0, 138, 255),       # Màu xanh dương sáng
        "index_finger": (0, 217, 255), # Màu xanh lục
        "middle_finger": (0, 255, 255), # Màu vàng
        "ring_finger": (255, 191, 0),   # Màu cam
        "little_finger": (255, 0, 0),   # Màu đỏ
        "palm": (128, 128, 128),        # Màu xám
        "landmark": (255, 255, 255)     # Màu trắng cho điểm mốc
    }

    # Định nghĩa các ngón tay và vẽ
    fingers = [
        (2, 3, 4),       # Thumb
        (5, 6, 7, 8),    # Index finger
        (9, 10, 11, 12), # Middle finger
        (13, 14, 15, 16),# Ring finger
        (17, 18, 19, 20) # Little finger
    ]

    finger_names = ["thumb", "index_finger", "middle_finger", "ring_finger", "little_finger"]

    for i, finger in enumerate(fingers):
        color = mp_colors[finger_names[i]]  # Lấy màu chuẩn từ Mediapipe
        for j in range(len(finger) - 1):
            draw_line(image, landmark_point[finger[j]], landmark_point[finger[j + 1]], color, 4)

    # Vẽ lòng bàn tay
    palm_connections = [
        (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)
    ]
    for connection in palm_connections:
        draw_line(image, landmark_point[connection[0]], landmark_point[connection[1]], mp_colors["palm"], 4)

    # Vẽ các điểm mốc
    for index, landmark in enumerate(landmark_point):
        radius = 8 if index % 4 == 0 else 5  # Điểm đầu ngón to hơn
        draw_circle(image, landmark, radius, mp_colors["landmark"])

    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle with a thicker border and a modern color (e.g., light blue)
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (255, 165, 0), 3)  # Orange with thicker border for visibility

    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    # Adding a translucent background for text
    overlay = image.copy()
    cv.rectangle(overlay, (brect[0], brect[1] - 22), (brect[2], brect[1]),
                 (0, 0, 0), -1)  # Black background
    alpha = 0.5  # Transparency level
    cv.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Preparing the text for handedness and gestures
    info_text = handedness.classification[0].label
    if hand_sign_text:
        info_text += f': {hand_sign_text}'

    # Set font properties with better readability
    font = cv.FONT_HERSHEY_COMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_color = (255, 255, 255)  # White text

    # Adding a drop shadow for better text visibility
    shadow_offset = 2
    cv.putText(image, info_text, (brect[0] + 5 + shadow_offset, brect[1] - 4 + shadow_offset),
               font, font_scale, (0, 0, 0), font_thickness + 2, cv.LINE_AA)  # Shadow in black

    # Main text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               font, font_scale, text_color, font_thickness, cv.LINE_AA)

    # Uncomment and add finger gesture text if needed
    # if finger_gesture_text:
    #     finger_gesture_label = f"Finger Gesture: {finger_gesture_text}"
    #     cv.putText(image, finger_gesture_label, (10, 60),
    #                font, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    return image

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image

def draw_info(image, fps, mode, number):
    # Prepare background for FPS to improve readability
    overlay = image.copy()
    cv.rectangle(overlay, (5, 5), (150, 50), (0, 0, 0), -1)  # Black background for FPS text
    alpha = 0.6  # Transparency
    cv.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # FPS text with drop shadow
    cv.putText(image, f"FPS: {fps}", (10 + 2, 30 + 2), cv.FONT_HERSHEY_SIMPLEX, 
               1.0, (0, 0, 0), 4, cv.LINE_AA)  # Shadow in black
    cv.putText(image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
               1.0, (255, 255, 255), 2, cv.LINE_AA)  # Main text in white

    # Mode info with clean background and shadow effect
    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, f"MODE: {mode_string[mode - 1]}", (10 + 2, 90 + 2),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)  # Shadow
        cv.putText(image, f"MODE: {mode_string[mode - 1]}", (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        # Display the number if it's within the valid range
        if 0 <= number <= 9:
            cv.putText(image, f"NUM: {number}", (10 + 2, 110 + 2),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)  # Shadow
            cv.putText(image, f"NUM: {number}", (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

def draw_recognized_text(image, text_list):
    # Cập nhật văn bản bằng cách lấy từ mới nhất
    text = " ".join(text_list[-5:])  # Display the last 10 words
     
    # Tạo một bản sao của ảnh để làm nền cho văn bản
    overlay = image.copy()
    
    # Vẽ một hình chữ nhật nền cho văn bản
    cv.rectangle(overlay, (5, 40), (image.shape[1] - 5, 90), (0, 0, 0), -1)
    
    # Cài đặt độ mờ cho nền
    alpha = 0.7
    cv.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Thêm hiệu ứng bóng đổ cho văn bản
    cv.putText(image, f'Recognized: {text}', (10 + 2, 70 + 2),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv.LINE_AA)  # Shadow
    # Vẽ văn bản chính với màu xanh lá cây
    cv.putText(image, f'Recognized: {text}', (10, 70),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)  # Main text in green
    
    return image

def update_recognized_text(recognized_text, hand_sign_id, keypoint_classifier_labels):
    # Lấy từ mới nhận diện từ hand_sign_id
    new_text = keypoint_classifier_labels[hand_sign_id]
    
    # Kiểm tra xem từ mới có khác từ cuối cùng không
    if len(recognized_text) == 0 or recognized_text[-1] != new_text:
        # Nếu khác, thêm từ mới vào danh sách
        recognized_text.append(new_text)
    
    return recognized_text

if __name__ == '__main__':
    main()
