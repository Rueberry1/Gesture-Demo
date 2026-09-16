# to run this code:
# 1. install python 3.10+ on your computer at https://www.python.org/downloads/
# 2. ensure you have pip installed on your computer to download the libraries below
# 3. open a terminal or command prompt and type pip install and then each of the libraries below
#    mediapipe, opencv-python (this would be installed with mediapipe), os, csv, time, math, datetime, pathlib
# 4. run the code by typing cd and then the path of the folder containing this code, it is most likely C:\Users\YOUR-USERNAME\Downloads
# 5. then type python gesture_demo.py
# if any of the above steps do not work you may be on a different operating system, look up how to install python libraries and run scripts
import cv2
import mediapipe as mp
import os
import csv
import time
import math
from datetime import datetime
from pathlib import Path

# Use MediaPipe Holistic for full-body (pose + face + hands)
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

# Save directory (script directory by default)
BASE_DIR = Path(__file__).resolve().parent
save_dir = BASE_DIR
save_dir.mkdir(parents=True, exist_ok=True)

log_file = save_dir / "gesture_logs.csv"

# Create CSV file if it doesn’t exist
if not log_file.exists():
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["datetime", "gesture", "confidence"])

def log_gesture(gesture, confidence):
    """Save detected gestures with timestamp and confidence into CSV."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([now, gesture, round(confidence, 2)])


def landmark(pose_landmarks, idx):
    """Safe accessor for landmarks (returns None if missing)."""
    if not pose_landmarks:
        return None
    return pose_landmarks.landmark[idx]


def calculate_angle(p1, p2, p3):
    """Calculate angle between three points (p2 is the vertex) in degrees."""
    if not all([p1, p2, p3]):
        return None
    
    # Convert to vectors
    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    
    # Calculate dot product and magnitudes
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return None
    
    # Calculate angle in radians, then convert to degrees
    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to [-1, 1]
    angle = math.acos(cos_angle)
    return math.degrees(angle)


def put_text_outlined(frame, text, position, font, font_scale, color, thickness, outline_thickness=1):
    """Draw text with thin black outline using fast OpenCV rendering with anti-aliasing."""
    x, y = position
    
    # OpenCV uses baseline positioning - y is the baseline, text extends upward
    # Get text size to ensure proper positioning
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # For top-aligned text: if y represents the desired top position,
    # we need baseline = y + text_height (since baseline is at bottom of text)
    # Add small margin for outline
    y_adjusted = y + text_height + baseline + outline_thickness
    
    # Draw thin black outline first (much thinner for professional look)
    outline_color = (0, 0, 0)
    outline_thick = max(1, thickness + outline_thickness)  # Much thinner outline
    
    # Draw outline in 4 cardinal directions for subtle outline
    offsets = [(-outline_thickness, 0), (outline_thickness, 0), 
               (0, -outline_thickness), (0, outline_thickness)]
    
    for dx, dy in offsets:
        cv2.putText(frame, text, (x + dx, y_adjusted + dy), font, font_scale, 
                   outline_color, outline_thick, cv2.LINE_AA)
    
    # Draw the main text on top with anti-aliasing
    cv2.putText(frame, text, (x, y_adjusted), font, font_scale, color, thickness, cv2.LINE_AA)


def analyze_running_form(pose_landmarks):
    """
    Analyze running/sprinting form and return overall score and improvement tips.
    Returns: (form_score, improvement_tips_list)
    """
    if not pose_landmarks:
        return None, []
    
    improvement_tips = []
    scores = []
    
    # Get key landmarks
    l_shoulder = landmark(pose_landmarks, mp_holistic.PoseLandmark.LEFT_SHOULDER)
    r_shoulder = landmark(pose_landmarks, mp_holistic.PoseLandmark.RIGHT_SHOULDER)
    l_hip = landmark(pose_landmarks, mp_holistic.PoseLandmark.LEFT_HIP)
    r_hip = landmark(pose_landmarks, mp_holistic.PoseLandmark.RIGHT_HIP)
    l_knee = landmark(pose_landmarks, mp_holistic.PoseLandmark.LEFT_KNEE)
    r_knee = landmark(pose_landmarks, mp_holistic.PoseLandmark.RIGHT_KNEE)
    l_ankle = landmark(pose_landmarks, mp_holistic.PoseLandmark.LEFT_ANKLE)
    r_ankle = landmark(pose_landmarks, mp_holistic.PoseLandmark.RIGHT_ANKLE)
    l_elbow = landmark(pose_landmarks, mp_holistic.PoseLandmark.LEFT_ELBOW)
    r_elbow = landmark(pose_landmarks, mp_holistic.PoseLandmark.RIGHT_ELBOW)
    l_wrist = landmark(pose_landmarks, mp_holistic.PoseLandmark.LEFT_WRIST)
    r_wrist = landmark(pose_landmarks, mp_holistic.PoseLandmark.RIGHT_WRIST)
    nose = landmark(pose_landmarks, mp_holistic.PoseLandmark.NOSE)
    
    # 1. KNEE DRIVE ANGLE (most important for sprinting)
    knee_scores = []
    knee_feedback = []
    if l_hip and l_knee and l_ankle:
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        if l_knee_angle:
            # Ideal knee drive angle: 120-150° when knee is up
            if l_knee.y < l_hip.y:  # Knee is raised
                ideal_angle = 130
                if l_knee_angle >= 110:
                    score = min(100, (l_knee_angle / ideal_angle) * 100)
                else:
                    score = max(0, (l_knee_angle / 110) * 50)
                knee_scores.append(score)
                knee_feedback.append(("Left", l_knee_angle, score))
    
    if r_hip and r_knee and r_ankle:
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        if r_knee_angle:
            if r_knee.y < r_hip.y:  # Knee is raised
                ideal_angle = 130
                if r_knee_angle >= 110:
                    score = min(100, (r_knee_angle / ideal_angle) * 100)
                else:
                    score = max(0, (r_knee_angle / 110) * 50)
                knee_scores.append(score)
                knee_feedback.append(("Right", r_knee_angle, score))
    
    if knee_scores:
        avg_knee_score = sum(knee_scores) / len(knee_scores)
        scores.append(avg_knee_score * 0.25)  # 25% weight
        
        # Add improvement tip for worst knee
        if knee_feedback:
            worst_knee = min(knee_feedback, key=lambda x: x[2])
            if worst_knee[2] < 70:
                tip = f"Knee Drive: {worst_knee[2]:.0f}%, increase the angle of your {worst_knee[0].lower()} knee"
                improvement_tips.append(tip)
    
    # 2. ARM SWING (elbow angle should be ~90°)
    arm_scores = []
    arm_feedback = []
    if l_shoulder and l_elbow and l_wrist:
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        if l_elbow_angle:
            ideal_elbow = 90
            deviation = abs(l_elbow_angle - ideal_elbow)
            score = max(0, 100 - (deviation * 2))  # 2 points per degree off
            arm_scores.append(score)
            arm_feedback.append(("Left", l_elbow_angle, score))
    
    if r_shoulder and r_elbow and r_wrist:
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        if r_elbow_angle:
            ideal_elbow = 90
            deviation = abs(r_elbow_angle - ideal_elbow)
            score = max(0, 100 - (deviation * 2))
            arm_scores.append(score)
            arm_feedback.append(("Right", r_elbow_angle, score))
    
    if arm_scores:
        avg_arm_score = sum(arm_scores) / len(arm_scores)
        scores.append(avg_arm_score * 0.20)  # 20% weight
        
        if arm_feedback:
            worst_arm = min(arm_feedback, key=lambda x: x[2])
            if worst_arm[2] < 70:
                direction = "bend" if worst_arm[1] > 100 else "straighten"
                tip = f"Arm Swing: {worst_arm[2]:.0f}%, {direction} your {worst_arm[0].lower()} elbow more"
                improvement_tips.append(tip)
    
    # 3. BODY LEAN (forward lean angle)
    if nose and l_hip and r_hip:
        avg_hip_y = (l_hip.y + r_hip.y) / 2
        if l_ankle:
            lean_factor = (nose.x - (l_hip.x + r_hip.x) / 2)
            body_height = abs(nose.y - avg_hip_y)
            if body_height > 0:
                lean_angle_approx = math.degrees(math.atan(lean_factor / body_height))
                ideal_lean = 8  # degrees forward
                deviation = abs(lean_angle_approx - ideal_lean)
                lean_score = max(0, 100 - (deviation * 5))
                scores.append(lean_score * 0.15)  # 15% weight
                
                if lean_score < 70:
                    direction = "lean forward more" if lean_angle_approx < ideal_lean else "reduce forward lean"
                    tip = f"Body Lean: {lean_score:.0f}%, {direction}"
                    improvement_tips.append(tip)
    
    # 4. LEG EXTENSION (during push-off phase)
    extension_scores = []
    extension_feedback = []
    if l_hip and l_knee and l_ankle:
        l_leg_angle = calculate_angle(l_hip, l_knee, l_ankle)
        if l_leg_angle:
            if l_ankle.x > l_knee.x:  # Leg is extended back
                ideal_extension = 170
                if l_leg_angle >= 150:
                    score = min(100, ((l_leg_angle - 150) / (ideal_extension - 150)) * 100)
                else:
                    score = max(0, (l_leg_angle / 150) * 50)
                extension_scores.append(score)
                extension_feedback.append(("Left", l_leg_angle, score))
    
    if r_hip and r_knee and r_ankle:
        r_leg_angle = calculate_angle(r_hip, r_knee, r_ankle)
        if r_leg_angle:
            if r_ankle.x > r_knee.x:  # Leg is extended back
                ideal_extension = 170
                if r_leg_angle >= 150:
                    score = min(100, ((r_leg_angle - 150) / (ideal_extension - 150)) * 100)
                else:
                    score = max(0, (r_leg_angle / 150) * 50)
                extension_scores.append(score)
                extension_feedback.append(("Right", r_leg_angle, score))
    
    if extension_scores:
        avg_extension = sum(extension_scores) / len(extension_scores)
        scores.append(avg_extension * 0.20)  # 20% weight
        
        if extension_feedback:
            worst_leg = min(extension_feedback, key=lambda x: x[2])
            if worst_leg[2] < 70:
                tip = f"Leg Extension: {worst_leg[2]:.0f}%, extend your {worst_leg[0].lower()} leg further back"
                improvement_tips.append(tip)
    
    # 5. SHOULDER LEVEL (shoulders should be roughly level)
    if l_shoulder and r_shoulder:
        shoulder_diff = abs(l_shoulder.y - r_shoulder.y)
        if shoulder_diff < 0.02:
            shoulder_score = 100
        else:
            shoulder_score = max(0, 100 - (shoulder_diff * 2000))
        scores.append(shoulder_score * 0.10)  # 10% weight
        
        if shoulder_score < 70:
            higher = "left" if l_shoulder.y < r_shoulder.y else "right"
            tip = f"Shoulder Level: {shoulder_score:.0f}%, level your shoulders ({higher} shoulder too high)"
            improvement_tips.append(tip)
    
    # 6. HIP ALIGNMENT (hips should be relatively level)
    if l_hip and r_hip:
        hip_diff = abs(l_hip.y - r_hip.y)
        if hip_diff < 0.02:
            hip_score = 100
        else:
            hip_score = max(0, 100 - (hip_diff * 2000))
        scores.append(hip_score * 0.10)  # 10% weight
        
        if hip_score < 70:
            higher = "left" if l_hip.y < r_hip.y else "right"
            tip = f"Hip Alignment: {hip_score:.0f}%, align your hips ({higher} hip too high)"
            improvement_tips.append(tip)
    
    # Calculate overall form score
    if scores:
        form_score = sum(scores)
        form_score = max(0, min(100, form_score))  # Clamp to 0-100
        return form_score, improvement_tips
    
    return None, []


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Rate-limit logging (seconds)
    min_log_interval = 1.0
    last_log_time = 0.0
    last_logged_gesture = None

    # Use Holistic to get pose + hands + face
    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(rgb)

            detected_gesture = None
            confidence_score = 0.0

            # Draw landmarks on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Analyze running form
            form_score, improvement_tips = analyze_running_form(results.pose_landmarks)
            
            # Display form analysis
            if form_score is not None:
                # Main score display
                score_text = f"Form Score: {form_score:.1f}%"
                color = (0, 255, 0) if form_score >= 70 else (0, 165, 255) if form_score >= 50 else (0, 0, 255)
                put_text_outlined(frame, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                # Display improvement tips (use detected_gesture for the primary tip)
                y_offset = 70
                max_tips = 3  # Show max 3 tips to avoid cluttering
                
                for i, tip in enumerate(improvement_tips[:max_tips]):
                    # Use detected_gesture for the most important tip (first one)
                    if i == 0:
                        detected_gesture = tip
                        confidence_score = form_score / 100
                    
                    # Display all tips
                    put_text_outlined(frame, tip, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25
                
                # Score bar
                bar_x, bar_y, bar_w, bar_h = 10, frame.shape[0] - 60, 300, 30
                filled_w = int(bar_w * (form_score / 100))
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), color, -1)
                
                # Rate-limited logging
                now = time.time()
                if (now - last_log_time) >= min_log_interval:
                    log_gesture(f"Form_{form_score:.1f}%", form_score / 100)
                    if detected_gesture:
                        last_logged_gesture = detected_gesture
                    last_log_time = now
            else:
                put_text_outlined(frame, "No pose detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display quit instruction (always visible, bottom right)
            quit_text = "Press 'q' to quit"
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(quit_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            # Position from right edge with padding
            text_x = frame.shape[1] - text_width - 20
            # Position from bottom - pass the desired top position (the function will adjust for baseline)
            # We want text bottom at frame bottom minus padding, so top = frame_bottom - padding - text_height - baseline
            text_y_top = frame.shape[0] - 15 - text_height - baseline  # 15px padding from bottom
            put_text_outlined(frame, quit_text, (text_x, text_y_top), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            cv2.imshow("Track & Field Form Analysis", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
