# main.py
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# --- Global Setup ---

# This will store the points in our drawing
points = deque(maxlen=128)

# Initialize MediaPipe Hands solution
mp_hands = mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# --- New: Landmark indices for tips and "pip" joints (second joint from tip) ---
# We use these to check if a finger is extended
finger_tip_ids = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                  mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                  mp_hands.HandLandmark.PINKY_TIP]

finger_pip_ids = [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
                  mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP,
                  mp_hands.HandLandmark.PINKY_PIP]

# --- New: Smoothing variables for jitter reduction ---
prev_x, prev_y = 0, 0
smooth_factor = 0.5 # Adjust this (0.1 = more smooth, 0.9 = more responsive)

# Start the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Set width
cap.set(4, 720)  # Set height

print("Starting Air-Writer v0.2...")
print("- Show ONE (index) finger to write.")
print("- Show FIVE (full palm) fingers to classify & clear.")
print("- Press 'c' to clear manually. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    canvas = np.zeros_like(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0] # Get the first (and only) hand
        lm = hand_landmarks.landmark # Shorthand for landmarks

        # --- New: Finger counting logic ---
        fingers_up = [False] * 5
        
        # Check Thumb: Compare tip.x to pip.x (for horizontal extension)
        # Assumes a right hand, adjust if < or > is wrong for you
        if lm[finger_tip_ids[0]].x < lm[finger_pip_ids[0]].x:
             fingers_up[0] = True
        
        # Check other 4 fingers: Compare tip.y to pip.y (for vertical extension)
        for i in range(1, 5):
            if lm[finger_tip_ids[i]].y < lm[finger_pip_ids[i]].y:
                fingers_up[i] = True
        
        total_fingers_up = sum(fingers_up)
        # --- End of new logic ---


        # --- Modified: State Machine (Write, Classify, Standby) ---
        
        # State 1: "Write Mode" (Only Index finger is up)
        if fingers_up[1] and not fingers_up[0] and not fingers_up[2] and not fingers_up[3] and not fingers_up[4]:
            
            # Get index finger tip coordinate
            h, w, c = frame.shape
            cx, cy = int(lm[finger_tip_ids[1]].x * w), int(lm[finger_tip_ids[1]].y * h)

            # --- New: Smoothing logic ---
            if prev_x == 0 and prev_y == 0: # First frame
                prev_x, prev_y = cx, cy
                
            smooth_cx = int(prev_x * (1 - smooth_factor) + cx * smooth_factor)
            smooth_cy = int(prev_y * (1 - smooth_factor) + cy * smooth_factor)

            # Add the smoothed point to our deque
            points.append((smooth_cx, smooth_cy))
            
            # Update previous points
            prev_x, prev_y = smooth_cx, smooth_cy
            
            # Draw a circle on the smoothed fingertip position
            cv2.circle(frame, (smooth_cx, smooth_cy), 10, (0, 255, 0), cv2.FILLED)
        
        # State 2: "Classify Mode" (All 5 fingers are up)
        elif all(fingers_up):
            # This is where we will trigger the model later
            # For now, we print to console and clear the drawing
            print("CLASSIFYING!") # Placeholder for model inference
            
            # --- We will soon save the 'points' data here ---
            
            points.clear()
            prev_x, prev_y = 0, 0 # Reset smoother
        
        # State 3: "Standby Mode" (Anything else)
        else:
            # "Lift the pen" by adding None, which breaks the line
            points.append(None)
            prev_x, prev_y = 0, 0 # Reset smoother

    else:
        # No hand found, "lift the pen"
        points.append(None)
        prev_x, prev_y = 0, 0 # Reset smoother

    # --- Drawing Logic (now handles 'None' for pen lifts) ---
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(canvas, points[i - 1], points[i], (255, 0, 0), 5) # Blue trail

    frame = cv2.addWeighted(frame, 1.0, canvas, 0.8, 0)
    cv2.imshow("Air-Writer v0.2", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        points.clear()
        prev_x, prev_y = 0, 0

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
hands.close()