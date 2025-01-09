import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b) 
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

def check_curl_form(shoulder, elbow, wrist, hip):
    # Check if elbow is kept close to body
    elbow_distance = abs(elbow[0] - hip[0])
    
    # Check if shoulder stays still
    shoulder_movement = shoulder[1]  # Vertical position
    
    form_feedback = []
    if elbow_distance > 0.1:  # Threshold value
        form_feedback.append("Keep elbow close to body")
    
    if shoulder_movement < 0.2:  # Threshold value
        form_feedback.append("Keep shoulder still")
        
    return form_feedback

def check_squat_form(hip, knee, ankle, shoulder):
    # Check knee alignment
    knee_x = knee[0]
    ankle_x = ankle[0]
    knee_ankle_alignment = abs(knee_x - ankle_x)
    
    # Check back angle
    back_angle = calculate_angle(shoulder, hip, knee)
    
    form_feedback = []
    if knee_ankle_alignment > 0.1:  # Threshold value
        form_feedback.append("Knees over toes")
        
    if back_angle < 45:  # Threshold value
        form_feedback.append("Keep back straight")
        
    return form_feedback

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    curl_counter = 0
    squat_counter = 0
    curl_stage = None
    squat_stage = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Calculate angles
            curl_angle = calculate_angle(shoulder, elbow, wrist)
            squat_angle = calculate_angle(hip, knee, ankle)
            
            # Check form
            curl_feedback = check_curl_form(shoulder, elbow, wrist, hip)
            squat_feedback = check_squat_form(hip, knee, ankle, shoulder)
            
            # Counter logic
            if curl_angle > 160:
                curl_stage = "down"
            if curl_angle < 30 and curl_stage == "down":
                curl_stage = "up"
                curl_counter += 1
                
            if squat_angle > 160:
                squat_stage = "up"
            if squat_angle < 90 and squat_stage == "up":
                squat_stage = "down"
                squat_counter += 1
            
            # Display form feedback
            y_position = 160
            for feedback in curl_feedback:
                cv2.putText(image, feedback, (10, y_position),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                y_position += 30
                
            for feedback in squat_feedback:
                cv2.putText(image, feedback, (10, y_position),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                y_position += 30
                
        except:
            pass
        
        # Display counters and stages
        cv2.putText(image, f'Curls: {curl_counter}', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(image, f'Curl Stage: {curl_stage}', (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(image, f'Squats: {squat_counter}', (10,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(image, f'Squat Stage: {squat_stage}', (10,130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Exercise Form Tracker', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
