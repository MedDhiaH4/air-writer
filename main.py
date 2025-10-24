# main.py

# --- 1. Imports ---
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from preprocessing import preprocess_image  # Import our processing function

# --- 2. Global Setup & Constants ---

points = deque(maxlen=128)
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7,
                       max_num_hands=1)

finger_tip_ids = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                  mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                  mp_hands.HandLandmark.PINKY_TIP]

finger_pip_ids = [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
                  mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP,
                  mp_hands.HandLandmark.PINKY_PIP]

prev_x, prev_y = 0, 0
smooth_factor = 0.5
last_gesture = "standby"

# --- 3. Webcam Initialization ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("Starting Air-Writer v0.7...")
print("- Show ONE (index) finger to write.")
print("- Show FIVE (full palm) fingers to classify & clear.")
print("- Show ONE (pinky) finger to delete/clear.")
print("- Press 'c' to clear manually. Press 'q' to quit.")

# --- 4. Main Application Loop ---
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # --- 5. Hand Landmark & Gesture Logic ---
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark

        # --- Finger Counting Logic ---
        fingers_up = [False] * 5
        if lm[finger_tip_ids[0]].x < lm[finger_pip_ids[0]].x:
             fingers_up[0] = True
        for i in range(1, 5):
            if lm[finger_tip_ids[i]].y < lm[finger_pip_ids[i]].y:
                fingers_up[i] = True
        
        # --- State Machine (Write, Classify, Delete, Standby) ---
        
        # Determine the CURRENT gesture
        current_gesture = "standby"  # Default
        if fingers_up[1] and not fingers_up[0] and not fingers_up[2] and not fingers_up[3] and not fingers_up[4]:
            current_gesture = "write"
        elif all(fingers_up):
            current_gesture = "classify"
        # Check for "pinky up" gesture
        elif fingers_up[4] and not fingers_up[0] and not fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
            current_gesture = "delete"
        
        # Now, act based on the current gesture AND the last gesture
        
        # State 1: "Write Mode"
        if current_gesture == "write":
            h, w, c = frame.shape
            cx, cy = int(lm[finger_tip_ids[1]].x * w), int(lm[finger_tip_ids[1]].y * h)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = cx, cy
            
            smooth_cx = int(prev_x * (1 - smooth_factor) + cx * smooth_factor)
            smooth_cy = int(prev_y * (1 - smooth_factor) + cy * smooth_factor)

            points.append((smooth_cx, smooth_cy))
            prev_x, prev_y = smooth_cx, smooth_cy
            
            cv2.circle(frame, (smooth_cx, smooth_cy), 10, (0, 0, 0), cv2.FILLED)
        
        # State 2: "Classify Mode"
        elif current_gesture == "classify" and last_gesture != "classify":
            # This block runs on the *first* frame the palm is shown
            print("CLASSIFYING!")
            
            # Preprocess the canvas to get the 64x64 letter
            processed_img = preprocess_image(canvas, size=64)
            
            # Resize the 64x64 image to 256x256 for a clear display
            display_img = cv2.resize(processed_img, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Model Input (Processed 64x64)", display_img)
            
            # Reset everything for the next letter
            points.clear()
            canvas.fill(0)
            prev_x, prev_y = 0, 0
        
        # State 3: "Delete Mode"
        elif current_gesture == "delete" and last_gesture != "delete":
            # This block runs on the *first* frame the pinky is shown
            print("DELETE!")
            # For now, this just clears the current canvas
            # Later, this will pop from our list of recognized letters
            points.clear()
            canvas.fill(0)
            prev_x, prev_y = 0, 0

        # State 4: "Standby Mode"
        elif current_gesture == "standby":
            points.append(None) # "Lift the pen"
            prev_x, prev_y = 0, 0
        
        # Update the last_gesture at the end of the frame
        last_gesture = current_gesture

    # --- 6. No Hand Found Logic ---
    else:
        points.append(None)
        prev_x, prev_y = 0, 0
        last_gesture = "standby"

    # --- 7. Drawing Logic ---
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        
        cv2.line(frame, points[i - 1], points[i], (0, 0, 0), 5)
        cv2.line(canvas, points[i - 1], points[i], (255, 255, 255), 10)

    # --- 8. Display Windows ---
    cv2.imshow("Air-Writer v0.7 (User View)", frame)

    # --- 9. User Controls ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        points.clear()
        canvas.fill(0)
        prev_x, prev_y = 0, 0
        last_gesture = "standby"

# --- 10. Cleanup ---
cap.release()
cv2.destroyAllWindows()
hands.close()