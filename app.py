from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import math
import numpy as np
import time

app = Flask(__name__)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Global variables for state management
capture_state = "front"  # "front", "side", "complete"
front_measurements = None
side_measurements = None
start_time = None
countdown = 10
final_body_shape = "Processing..."
all_measurements = {}

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def calculate_body_measurements(landmarks, image_shape, view_type="front"):
    # Get key landmarks
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    
    measurements = {}
    points = {}
    
    if view_type == "front":
        # Calculate shoulder width
        shoulder_width = get_distance(left_shoulder, right_shoulder)
        points['left_shoulder'] = (int(left_shoulder.x * image_shape[1]), int(left_shoulder.y * image_shape[0]))
        points['right_shoulder'] = (int(right_shoulder.x * image_shape[1]), int(right_shoulder.y * image_shape[0]))
        
        # Calculate hip width
        hip_width = get_distance(left_hip, right_hip)
        points['left_hip'] = (int(left_hip.x * image_shape[1]), int(left_hip.y * image_shape[0]))
        points['right_hip'] = (int(right_hip.x * image_shape[1]), int(right_hip.y * image_shape[0]))
        
        # Estimate bust width (using shoulders and adjusting for chest)
        bust_width = shoulder_width * 1.15  # Approximation
        # Bust point is slightly below shoulders
        bust_y = (left_shoulder.y + right_shoulder.y) / 2 + 0.02
        bust_x = (left_shoulder.x + right_shoulder.x) / 2
        points['bust'] = (int(bust_x * image_shape[1]), int(bust_y * image_shape[0]))
        
        # Estimate waist width (using torso landmarks)
        mid_torso_y = (left_shoulder.y + left_hip.y) / 2
        left_waist = find_closest_landmark_at_y(landmarks, mid_torso_y, True)
        right_waist = find_closest_landmark_at_y(landmarks, mid_torso_y, False)
        waist_width = get_distance(left_waist, right_waist) if left_waist and right_waist else 0
        points['left_waist'] = (int(left_waist.x * image_shape[1]), int(left_waist.y * image_shape[0])) if left_waist else None
        points['right_waist'] = (int(right_waist.x * image_shape[1]), int(right_waist.y * image_shape[0])) if right_waist else None
        
        # High hip (midway between waist and hip)
        if left_waist and right_waist:
            high_hip_y = (left_waist.y + left_hip.y) / 2
            high_hip_x_left = left_waist.x + (left_hip.x - left_waist.x) * 0.5
            high_hip_x_right = right_waist.x + (right_hip.x - right_waist.x) * 0.5
            high_hip_width = get_distance(
                type('obj', (object,), {'x': high_hip_x_left, 'y': high_hip_y})(),
                type('obj', (object,), {'x': high_hip_x_right, 'y': high_hip_y})()
            )
            points['left_high_hip'] = (int(high_hip_x_left * image_shape[1]), int(high_hip_y * image_shape[0]))
            points['right_high_hip'] = (int(high_hip_x_right * image_shape[1]), int(high_hip_y * image_shape[0]))
        else:
            high_hip_width = 0
            points['left_high_hip'] = None
            points['right_high_hip'] = None
        
        # Calculate torso height
        torso_height = abs(left_shoulder.y - left_hip.y)
        
        # Calculate leg length
        leg_length = abs(left_hip.y - left_ankle.y)
        
        # Calculate ratios
        waist_to_hip = waist_width / hip_width if hip_width > 0 else 0
        bust_to_hip = bust_width / hip_width if hip_width > 0 else 0
        shoulder_to_hip = shoulder_width / hip_width if hip_width > 0 else 0
        torso_to_leg = torso_height / leg_length if leg_length > 0 else 0
        
        measurements = {
            'shoulder_width': shoulder_width,
            'bust_width': bust_width,
            'waist_width': waist_width,
            'hip_width': hip_width,
            'high_hip_width': high_hip_width,
            'torso_height': torso_height,
            'leg_length': leg_length,
            'waist_to_hip_ratio': waist_to_hip,
            'bust_to_hip_ratio': bust_to_hip,
            'shoulder_to_hip_ratio': shoulder_to_hip,
            'torso_to_leg_ratio': torso_to_leg
        }
    
    elif view_type == "side":
        # Calculate body depth measurements from side view
        # Use the midpoint between shoulders and hips as reference
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_mid_x = (left_hip.x + right_hip.x) / 2
        
        # Calculate chest depth (distance from front to back at chest level)
        chest_depth = abs(shoulder_mid_x - landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x)
        points['chest_front'] = (int(shoulder_mid_x * image_shape[1]), int(left_shoulder.y * image_shape[0]))
        points['chest_back'] = (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_shape[1]), int(left_shoulder.y * image_shape[0]))
        
        # Calculate waist depth (distance from front to back at waist level)
        waist_depth = abs(hip_mid_x - landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x) * 0.9
        waist_y = (left_shoulder.y + left_hip.y) / 2
        points['waist_front'] = (int(hip_mid_x * image_shape[1]), int(waist_y * image_shape[0]))
        points['waist_back'] = (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_shape[1]), int(waist_y * image_shape[0]))
        
        # Calculate hip depth
        hip_depth = abs(hip_mid_x - landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x)
        points['hip_front'] = (int(hip_mid_x * image_shape[1]), int(left_hip.y * image_shape[0]))
        points['hip_back'] = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * image_shape[1]), int(left_hip.y * image_shape[0]))
        
        # Calculate back curvature (angle between shoulder, hip, and ankle)
        back_angle = get_angle(left_shoulder, left_hip, left_ankle)
        
        # Calculate abdomen protrusion (distance from waist to hip line)
        abdomen_protrusion = abs(left_hip.y - waist_y) if 'waist_y' in locals() else 0
        
        measurements = {
            'chest_depth': chest_depth,
            'waist_depth': waist_depth,
            'hip_depth': hip_depth,
            'back_angle': back_angle,
            'abdomen_protrusion': abdomen_protrusion
        }
    
    return measurements, points

def find_closest_landmark_at_y(landmarks, target_y, left_side=True):
    closest_landmark = None
    min_distance = float('inf')
    
    relevant_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP
    ]
    
    for idx in relevant_landmarks:
        landmark = landmarks[idx]
        distance = abs(landmark.y - target_y)
        
        if left_side and landmark.x < 0.5:
            if distance < min_distance:
                min_distance = distance
                closest_landmark = landmark
        elif not left_side and landmark.x >= 0.5:
            if distance < min_distance:
                min_distance = distance
                closest_landmark = landmark
                
    return closest_landmark

def detect_body_shape(front_measurements, side_measurements):
    # Extract front measurements
    waist_to_hip = front_measurements['waist_to_hip_ratio']
    bust_to_hip = front_measurements['bust_to_hip_ratio']
    shoulder_to_hip = front_measurements['shoulder_to_hip_ratio']
    torso_to_leg = front_measurements['torso_to_leg_ratio']
    
    # Extract side measurements
    chest_depth = side_measurements['chest_depth']
    waist_depth = side_measurements['waist_depth']
    hip_depth = side_measurements['hip_depth']
    back_angle = side_measurements['back_angle']
    abdomen_protrusion = side_measurements['abdomen_protrusion']
    
    # Calculate depth ratios
    waist_to_chest = waist_depth / chest_depth if chest_depth > 0 else 0
    hip_to_chest = hip_depth / chest_depth if chest_depth > 0 else 0
    
    # Define thresholds for body shape classification
    hourglass_waist_threshold = 0.75
    significant_difference = 0.1
    moderate_difference = 0.05
    
    # Enhanced classification with side view data
    # Hourglass: Bust and hips similar, waist significantly smaller, balanced depth
    if (waist_to_hip < hourglass_waist_threshold and 
        abs(bust_to_hip - 1.0) < moderate_difference and
        abs(waist_to_chest - 0.85) < 0.1 and
        abs(hip_to_chest - 1.0) < 0.1):
        return "Hourglass"
    
    # Pear: Hips wider than bust and shoulders, defined waist, larger hip depth
    elif (bust_to_hip < 0.9 and 
          shoulder_to_hip < 0.9 and
          waist_to_hip < hourglass_waist_threshold and
          hip_to_chest > 1.1):
        return "Pear"
    
    # Apple: Bust wider than hips, waist less defined, larger abdomen protrusion
    elif (bust_to_hip > 1.1 and 
          waist_to_hip > 0.85 and
          abdomen_protrusion > 0.05 and
          waist_to_chest > 0.9):
        return "Apple"
    
    # Rectangle: Bust, waist, hips similar, balanced depth
    elif (abs(bust_to_hip - 1.0) < moderate_difference and
          abs(waist_to_hip - 1.0) < moderate_difference and
          abs(waist_to_chest - 0.9) < 0.1 and
          abs(hip_to_chest - 0.95) < 0.1):
        return "Rectangle"
    
    # Inverted Triangle: Shoulders wider than hips, larger chest depth
    elif (shoulder_to_hip > 1.1 and 
          bust_to_hip > 1.05 and
          waist_to_chest < 0.85):
        return "Inverted Triangle"
    
    # Athletic: Similar to rectangle but with more muscular build, straight back
    elif (abs(bust_to_hip - 1.0) < moderate_difference and
          abs(waist_to_hip - 1.0) < moderate_difference and
          torso_to_leg < 0.9 and
          back_angle > 170):
        return "Athletic"
    
    # Default case
    else:
        return "Unknown"

def draw_body_points(frame, points, view_type):
    # Define colors for different points
    colors = {
        'shoulder': (255, 0, 0),      # Red
        'bust': (0, 255, 0),         # Green
        'waist': (0, 0, 255),        # Blue
        'hip': (255, 255, 0),        # Cyan
        'high_hip': (255, 0, 255),   # Magenta
        'chest': (0, 255, 255),      # Yellow
        'back': (128, 0, 128),       # Purple
        'front': (0, 128, 255)       # Orange
    }
    
    # Draw front view points
    if view_type == "front":
        # Shoulders
        if 'left_shoulder' in points and points['left_shoulder']:
            cv2.circle(frame, points['left_shoulder'], 8, colors['shoulder'], -1)
            cv2.putText(frame, "L Shoulder", (points['left_shoulder'][0] + 10, points['left_shoulder'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['shoulder'], 1)
        if 'right_shoulder' in points and points['right_shoulder']:
            cv2.circle(frame, points['right_shoulder'], 8, colors['shoulder'], -1)
            cv2.putText(frame, "R Shoulder", (points['right_shoulder'][0] + 10, points['right_shoulder'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['shoulder'], 1)
        
        # Bust
        if 'bust' in points and points['bust']:
            cv2.circle(frame, points['bust'], 8, colors['bust'], -1)
            cv2.putText(frame, "Bust", (points['bust'][0] + 10, points['bust'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['bust'], 1)
        
        # Waist
        if 'left_waist' in points and points['left_waist']:
            cv2.circle(frame, points['left_waist'], 8, colors['waist'], -1)
            cv2.putText(frame, "L Waist", (points['left_waist'][0] + 10, points['left_waist'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['waist'], 1)
        if 'right_waist' in points and points['right_waist']:
            cv2.circle(frame, points['right_waist'], 8, colors['waist'], -1)
            cv2.putText(frame, "R Waist", (points['right_waist'][0] + 10, points['right_waist'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['waist'], 1)
        
        # Hips
        if 'left_hip' in points and points['left_hip']:
            cv2.circle(frame, points['left_hip'], 8, colors['hip'], -1)
            cv2.putText(frame, "L Hip", (points['left_hip'][0] + 10, points['left_hip'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['hip'], 1)
        if 'right_hip' in points and points['right_hip']:
            cv2.circle(frame, points['right_hip'], 8, colors['hip'], -1)
            cv2.putText(frame, "R Hip", (points['right_hip'][0] + 10, points['right_hip'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['hip'], 1)
        
        # High Hips
        if 'left_high_hip' in points and points['left_high_hip']:
            cv2.circle(frame, points['left_high_hip'], 8, colors['high_hip'], -1)
            cv2.putText(frame, "L High Hip", (points['left_high_hip'][0] + 10, points['left_high_hip'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['high_hip'], 1)
        if 'right_high_hip' in points and points['right_high_hip']:
            cv2.circle(frame, points['right_high_hip'], 8, colors['high_hip'], -1)
            cv2.putText(frame, "R High Hip", (points['right_high_hip'][0] + 10, points['right_high_hip'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['high_hip'], 1)
    
    # Draw side view points
    elif view_type == "side":
        # Chest
        if 'chest_front' in points and points['chest_front']:
            cv2.circle(frame, points['chest_front'], 8, colors['chest'], -1)
            cv2.putText(frame, "Chest Front", (points['chest_front'][0] + 10, points['chest_front'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['chest'], 1)
        if 'chest_back' in points and points['chest_back']:
            cv2.circle(frame, points['chest_back'], 8, colors['back'], -1)
            cv2.putText(frame, "Chest Back", (points['chest_back'][0] + 10, points['chest_back'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['back'], 1)
        
        # Waist
        if 'waist_front' in points and points['waist_front']:
            cv2.circle(frame, points['waist_front'], 8, colors['waist'], -1)
            cv2.putText(frame, "Waist Front", (points['waist_front'][0] + 10, points['waist_front'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['waist'], 1)
        if 'waist_back' in points and points['waist_back']:
            cv2.circle(frame, points['waist_back'], 8, colors['back'], -1)
            cv2.putText(frame, "Waist Back", (points['waist_back'][0] + 10, points['waist_back'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['back'], 1)
        
        # Hip
        if 'hip_front' in points and points['hip_front']:
            cv2.circle(frame, points['hip_front'], 8, colors['hip'], -1)
            cv2.putText(frame, "Hip Front", (points['hip_front'][0] + 10, points['hip_front'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['hip'], 1)
        if 'hip_back' in points and points['hip_back']:
            cv2.circle(frame, points['hip_back'], 8, colors['back'], -1)
            cv2.putText(frame, "Hip Back", (points['hip_back'][0] + 10, points['hip_back'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['back'], 1)

def gen_frames():
    global capture_state, front_measurements, side_measurements, start_time, countdown, final_body_shape, all_measurements
    
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect body landmarks
        results = pose.process(frame_rgb)
        
        # Handle state transitions and countdown
        current_time = time.time()
        
        if capture_state == "front":
            if start_time is None:
                start_time = current_time
            
            elapsed = current_time - start_time
            countdown = max(0, 10 - int(elapsed))
            
            if countdown > 0:
                # Display countdown for front view
                cv2.putText(frame, f"Front View: {countdown}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Stand facing the camera", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Capture front measurements
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    front_measurements, front_points = calculate_body_measurements(landmarks, frame.shape, "front")
                    all_measurements['front'] = front_points
                
                # Transition to side view
                capture_state = "side"
                start_time = None
                countdown = 10
        
        elif capture_state == "side":
            if start_time is None:
                start_time = current_time
            
            elapsed = current_time - start_time
            countdown = max(0, 10 - int(elapsed))
            
            if countdown > 0:
                # Display countdown for side view
                cv2.putText(frame, f"Side View: {countdown}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Turn to your side", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Capture side measurements
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    side_measurements, side_points = calculate_body_measurements(landmarks, frame.shape, "side")
                    all_measurements['side'] = side_points
                
                # Calculate final body shape
                if front_measurements and side_measurements:
                    final_body_shape = detect_body_shape(front_measurements, side_measurements)
                
                # Transition to complete state
                capture_state = "complete"
                start_time = current_time
        
        elif capture_state == "complete":
            # Display final results
            cv2.putText(frame, f"Body Shape: {final_body_shape}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if front_measurements:
                measurements_text = (
                    f"Shoulder: {front_measurements['shoulder_width']:.2f} | "
                    f"Bust: {front_measurements['bust_width']:.2f} | "
                    f"Waist: {front_measurements['waist_width']:.2f} | "
                    f"Hip: {front_measurements['hip_width']:.2f}"
                )
                cv2.putText(frame, measurements_text, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if side_measurements:
                side_text = (
                    f"Chest Depth: {side_measurements['chest_depth']:.2f} | "
                    f"Waist Depth: {side_measurements['waist_depth']:.2f} | "
                    f"Hip Depth: {side_measurements['hip_depth']:.2f}"
                )
                cv2.putText(frame, side_text, (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Reset after 15 seconds
            if current_time - start_time > 15:
                capture_state = "front"
                front_measurements = None
                side_measurements = None
                all_measurements = {}
                start_time = None
                final_body_shape = "Processing..."
        
        # Draw pose landmarks if available
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Draw custom body points based on current state
            if capture_state == "front" and 'front' in all_measurements:
                draw_body_points(frame, all_measurements['front'], "front")
            elif capture_state == "side" and 'side' in all_measurements:
                draw_body_points(frame, all_measurements['side'], "side")
            elif capture_state == "complete":
                if 'front' in all_measurements:
                    draw_body_points(frame, all_measurements['front'], "front")
                if 'side' in all_measurements:
                    draw_body_points(frame, all_measurements['side'], "side")
        
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)