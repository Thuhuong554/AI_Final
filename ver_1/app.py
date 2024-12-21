from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
import torch

# Check PyTorch version and CUDA availability
print(torch.__version__)
print(torch.cuda.is_available())

app = Flask(__name__)

# Load the trained YOLO model
model = YOLO("mediapipe_tuning_asl_model.pt")

# Initialize Mediapipe Hands for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Store detected letters in a deque with a max length of 20
detected_letters = deque(maxlen=20)

# Map category IDs to letters (A-Z)
category_id_to_name = {i: chr(65 + i) for i in range(26)}

# Global variables to track the current detected sentence and previous landmarks
current_sentence = ""
prev_landmarks = None
change_threshold = 0.1  # Threshold to detect significant hand pose changes


# Function to crop the frame to the hand region
def crop_to_hand(frame, hand_landmarks):
    h, w, _ = frame.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0

    # Determine bounding box from hand landmarks
    for lm in hand_landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        x_min = min(x_min, cx)
        y_min = min(y_min, cy)
        x_max = max(x_max, cx)
        y_max = max(y_max, cy)

    # Add padding around the bounding box
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    # Crop the frame to the bounding box
    cropped_frame = frame[y_min:y_max, x_min:x_max]
    return cropped_frame, (x_min, y_min, x_max, y_max)


# Function to check if the hand pose has significantly changed
def has_hand_changed(new_landmarks):
    global prev_landmarks
    if prev_landmarks is None:
        prev_landmarks = new_landmarks
        return True  # First detection always processes

    # Calculate the average distance between current and previous landmarks
    total_change = 0
    for prev_lm, new_lm in zip(prev_landmarks, new_landmarks):
        total_change += ((prev_lm.x - new_lm.x) ** 2 + (prev_lm.y - new_lm.y) ** 2) ** 0.5

    avg_change = total_change / len(new_landmarks)
    if avg_change > change_threshold:
        prev_landmarks = new_landmarks
        return True

    return False


# Generator function to process video frames and make predictions
def gen_frames():
    global detected_letters, current_sentence
    cap = cv2.VideoCapture(0)  # Open the camera
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert the frame from BGR to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands using Mediapipe
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if the hand pose has significantly changed
                if has_hand_changed(hand_landmarks.landmark):
                    # Crop to the hand region
                    cropped_frame, _ = crop_to_hand(frame, hand_landmarks)

                    # Make predictions using YOLO on the cropped hand region
                    results = model.predict(cropped_frame, imgsz=640, conf=0.5, device='cpu')

                    # Process YOLO predictions
                    for box in results[0].boxes:
                        x_min, y_min, x_max, y_max = (
                            int(box.xyxy[0][0]),
                            int(box.xyxy[0][1]),
                            int(box.xyxy[0][2]),
                            int(box.xyxy[0][3]),
                        )
                        class_id = int(box.cls[0])  # Predicted class ID
                        letter = category_id_to_name[class_id]  # Map to letter
                        detected_letters.append(letter)
                        current_sentence = "".join(detected_letters)

                        # Draw bounding box and label on the original frame
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            letter,
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

        # Encode the frame for web streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


# Endpoint to serve the video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Endpoint to get the current detected sentence
@app.route('/get_sentence')
def get_sentence():
    return jsonify({"sentence": current_sentence})


# Endpoint to reset the detected sentence
@app.route('/reset', methods=['POST'])
def reset():
    global detected_letters, current_sentence
    detected_letters.clear()
    current_sentence = ""
    return '', 204


# Render the main HTML page
@app.route('/')
def index():
    return render_template('index.html')


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
