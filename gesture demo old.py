import cv2
import mediapipe as mp
import time
import os
import csv
from datetime import datetime

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face = mp_face.FaceMesh(min_detection_confidence=0.7)

# Directory for saving logs
save_dir = r"C:\Users\reube\OneDrive\Desktop\Python Projects\Gesture Demo"
os.makedirs(save_dir, exist_ok=True)  # creates folder if missing

log_file = os.path.join(save_dir, "gesture_logs.csv")

# Create CSV file if it doesn’t exist
if not os.path.isfile(log_file):
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["datetime", "gesture", "confidence"])

cap = cv2.VideoCapture(0)

def log_gesture(gesture, confidence):
    """Save detected gestures with timestamp and confidence into CSV."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([now, gesture, round(confidence, 2)])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand and face detection
    hand_results = hands.process(rgb)
    face_results = face.process(rgb)

    detected_gesture = None
    confidence_score = 0.0

    # ----- Hand detection -----
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Example: detect pinch gesture (thumb + index close)
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            dist = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5

            if dist < 0.05:  # tuned threshold
                detected_gesture = "Pinch"
                confidence_score = 0.9
            else:
                detected_gesture = "Hand Detected"
                confidence_score = 0.6

    # ----- Face detection -----
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face.FACEMESH_CONTOURS)

            # Example: detect frown (eyebrow distance)
            left_brow = face_landmarks.landmark[70]
            right_brow = face_landmarks.landmark[300]
            brow_dist = abs(left_brow.y - right_brow.y)

            if brow_dist < 0.02:
                detected_gesture = "Frown"
                confidence_score = 0.85
            else:
                if detected_gesture is None:  # don’t override hand
                    detected_gesture = "Neutral Face"
                    confidence_score = 0.7

    # ----- Draw overlays -----
    if detected_gesture:
        cv2.putText(frame, f"{detected_gesture} ({confidence_score:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        # Confidence bar
        bar_x, bar_y, bar_w, bar_h = 10, 50, 200, 20
        filled_w = int(bar_w * confidence_score)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + filled_w, bar_y + bar_h), (0, 255, 0), -1)

        # Log to CSV
        log_gesture(detected_gesture, confidence_score)

    cv2.imshow("Gesture + Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
