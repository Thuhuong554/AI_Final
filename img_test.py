import cv2
import mediapipe as mp

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Đọc ảnh
image_path = r'C:\path\to\your\image.jpg'  # Thay thế đường dẫn với ảnh của bạn
image = cv2.imread(image_path)

# Kiểm tra nếu ảnh đã được đọc đúng
if image is None:
    print("Failed to load image.")
else:
    # Chuyển đổi ảnh sang RGB (Mediapipe yêu cầu ảnh RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Xử lý ảnh với Mediapipe
    results = hands.process(image_rgb)

    # Kiểm tra nếu Mediapipe phát hiện bàn tay
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            print(f"Landmarks: {hand_landmarks}")
            
            # Vẽ các điểm landmark lên ảnh
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
        # Hiển thị ảnh có vẽ các landmark
        cv2.imshow("Hand landmarks", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No hands detected in the image.")

# Đóng Mediapipe
hands.close()
