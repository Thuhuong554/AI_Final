from flask import Flask, render_template, Response, jsonify, request
import cv2
from ultralytics import YOLO
import torch
import mediapipe as mp
import time

app = Flask(__name__)

# Load YOLOv8 model
model_path = "./models/yolov8.pt"  
model = YOLO(model_path)

# Webcam setup
camera = cv2.VideoCapture(0)

# MediaPipe setup for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,  # Higher confidence for reliable detection
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Global variables for predictions
predicted_label = None  # Current detected label (e.g., "A", "B")
predicted_sentence = []  # List to store sentence as detected letters
prediction_history = []  # History of predictions for smoothing

def detect_hands_and_predict(frame):
    """
    Detects hands in the frame, predicts using YOLO, and constructs sentences based on predictions.
    """
    global predicted_label, predicted_sentence, prediction_history

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)

    # If no hands are detected, return the original frame
    if not hand_results.multi_hand_landmarks:
        return frame

    # Process each detected hand
    for hand_landmarks in hand_results.multi_hand_landmarks:
        # Calculate bounding box for the hand
        x_min = min([lm.x for lm in hand_landmarks.landmark])
        x_max = max([lm.x for lm in hand_landmarks.landmark])
        y_min = min([lm.y for lm in hand_landmarks.landmark])
        y_max = max([lm.y for lm in hand_landmarks.landmark])

        # Convert normalized coordinates to pixel values
        h, w, _ = frame.shape
        x_min = int(x_min * w)
        x_max = int(x_max * w)
        y_min = int(y_min * h)
        y_max = int(y_max * h)

        # Crop the frame to the detected hand region
        hand_frame = frame[y_min:y_max, x_min:x_max]

        # Skip invalid hand frames
        if hand_frame.size == 0 or hand_frame.shape[0] < 50 or hand_frame.shape[1] < 50:
            continue

        # YOLO prediction for the cropped hand region
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = model.predict(hand_frame, conf=0.5, device=device)

        # Process YOLO results
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])  # Class index
                predicted_label = model.names[cls]  # Predicted label (e.g., "A", "B")

                # Add prediction to history and smooth using majority voting
                prediction_history.append(predicted_label)
                if len(prediction_history) > 10:  # Keep last 10 predictions
                    prediction_history.pop(0)
                most_common_prediction = max(set(prediction_history), key=prediction_history.count)

                # Append to the sentence if it is a new prediction
                if not predicted_sentence or predicted_sentence[-1] != most_common_prediction:
                    predicted_sentence.append(most_common_prediction)

        # Draw landmarks and bounding box on the frame
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return frame

def generate_frames():
    """
    Generates frames from the webcam for live video feed.
    """
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Process the frame for hand detection and prediction
        frame = detect_hands_and_predict(frame)

        # Encode the frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Provides the live video feed to the frontend."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    """Returns the current prediction and sentence."""
    global predicted_label, predicted_sentence
    return jsonify({
        'current_prediction': predicted_label,
        'sentence': ''.join(predicted_sentence)
    })

@app.route('/reset_prediction', methods=['POST'])
def reset_prediction():
    """Resets the predicted sentence."""
    global predicted_sentence
    predicted_sentence = []
    return jsonify({'status': 'success'})

if __name__ == "__main__":
    app.run(debug=True)
