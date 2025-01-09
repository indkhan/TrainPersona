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

def check_pushup_form(shoulder, elbow, wrist, hip, ankle):
    form_feedback = []
    
    # Check back alignment (hip shouldn't sag or pike)
    back_line = calculate_angle(shoulder, hip, ankle)
    if back_line < 165:
        form_feedback.append("Keep body straight")
    
    # Check elbow angle for proper depth
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    if elbow_angle > 90:
        form_feedback.append("Lower chest to ground")
    
    # Check hand position
    if abs(shoulder[0] - wrist[0]) > 0.3:
        form_feedback.append("Hands at shoulder width")
        
    return form_feedback

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    curl_counter = 0
    squat_counter = 0
    pushup_counter = 0
    curl_stage = None
    squat_stage = None
    pushup_stage = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get all coordinates
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
            pushup_angle = calculate_angle(shoulder, elbow, wrist)
            
            # Push-up counter logic
            if pushup_angle > 160:
                pushup_stage = "up"
            if pushup_angle < 90 and pushup_stage == "up":
                pushup_stage = "down"
                pushup_counter += 1
                
            # Get push-up form feedback
            pushup_feedback = check_pushup_form(shoulder, elbow, wrist, hip, ankle)
            
            # Previous exercise logic remains the same...
            
            # Display push-up counter and feedback
            cv2.putText(image, f'Push-ups: {pushup_counter}', (10,170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,165,0), 2)
            cv2.putText(image, f'Push-up Stage: {pushup_stage}', (10,200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,165,0), 2)
            
            # Display form feedback
            y_position = 230
            for feedback in pushup_feedback:
                cv2.putText(image, feedback, (10, y_position),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                y_position += 30
                
        except:
            pass
        
        # Display original exercise counters and stages
        cv2.putText(image, f'Curls: {curl_counter}', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(image, f'Squats: {squat_counter}', (10,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Complete Workout Tracker', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
