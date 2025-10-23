# main.py

# --- 1. Imports ---
# Import necessary libraries
import cv2  # OpenCV for webcam and drawing
import numpy as np  # NumPy for array operations (like our canvas)
import mediapipe as mp  # MediaPipe for hand tracking
from collections import deque  # Deque for efficiently storing a trail of points

# --- 2. Global Setup & Constants ---

# 'points' will store the (x, y) coordinates of our finger trail
# 'maxlen' limits the trail to 128 points, creating a "fading" effect
points = deque(maxlen=128)

# 'canvas' is our invisible, model-ready input.
# We'll draw a white line on this black background.
# Dimensions (720, 1280, 3) match our 720p webcam feed.
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Initialize MediaPipe's hand tracking solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,  # Don't detect if confidence is low
                       min_tracking_confidence=0.7,  # Don't track if confidence is low
                       max_num_hands=1)  # We only care about one hand

# These are constant landmark indices for specific parts of the hand
# We use them to check if a finger is "up" or "down"
finger_tip_ids = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                  mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                  mp_hands.HandLandmark.PINKY_TIP]

finger_pip_ids = [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
                  mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP,
                  mp_hands.HandLandmark.PINKY_PIP]

# Variables for smoothing the drawing line to reduce jitter
prev_x, prev_y = 0, 0
smooth_factor = 0.5  # 0 = max smoothing, 1 = no smoothing

# State variable to track the last gesture, preventing repeated triggers
last_gesture = "standby"

# --- 3. Webcam Initialization ---
cap = cv2.VideoCapture(0)  # Start video capture (webcam 0)
cap.set(3, 1280)  # Set width to 1280
cap.set(4, 720)   # Set height to 720

# Print instructions for the user
print("Starting Air-Writer v0.5...")
print("- Show ONE (index) finger to write.")
print("- Show FIVE (full palm) fingers to classify & clear.")
print("- Press 'c' to clear manually. Press 'q' to quit.")

# --- 4. Main Application Loop ---
while True:
    # Read a new frame from the webcam
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally (selfie view) so it feels natural
    frame = cv2.flip(frame, 1)

    # Convert the frame from BGR (OpenCV default) to RGB (MediaPipe requirement)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and find hand landmarks
    results = hands.process(frame_rgb)

    # --- 5. Hand Landmark & Gesture Logic ---
    if results.multi_hand_landmarks:
        # Get the first (and only) hand found
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark  # Shorthand

        # --- Finger Counting Logic ---
        fingers_up = [False] * 5  # List to store state of 5 fingers
        
        # Check Thumb (special case: uses x-axis)
        if lm[finger_tip_ids[0]].x < lm[finger_pip_ids[0]].x:
             fingers_up[0] = True
        
        # Check other 4 fingers (uses y-axis)
        for i in range(1, 5):
            if lm[finger_tip_ids[i]].y < lm[finger_pip_ids[i]].y:
                fingers_up[i] = True
        
        # --- State Machine (Write, Classify, Standby) ---
        
        # First, determine the CURRENT gesture
        current_gesture = "standby"  # Default
        if fingers_up[1] and not fingers_up[0] and not fingers_up[2] and not fingers_up[3] and not fingers_up[4]:
            current_gesture = "write"
        elif all(fingers_up):
            current_gesture = "classify"
        
        # Now, act based on the current gesture AND the last gesture
        
        # State 1: "Write Mode"
        if current_gesture == "write":
            h, w, c = frame.shape
            cx, cy = int(lm[finger_tip_ids[1]].x * w), int(lm[finger_tip_ids[1]].y * h)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = cx, cy
            
            # Apply smoothing
            smooth_cx = int(prev_x * (1 - smooth_factor) + cx * smooth_factor)
            smooth_cy = int(prev_y * (1 - smooth_factor) + cy * smooth_factor)

            points.append((smooth_cx, smooth_cy))
            prev_x, prev_y = smooth_cx, smooth_cy
            
            cv2.circle(frame, (smooth_cx, smooth_cy), 10, (0, 0, 0), cv2.FILLED)
        
        # State 2: "Classify Mode" (Triggered only ONCE)
        elif current_gesture == "classify" and last_gesture != "classify":
            # This block only runs on the *first* frame the palm is shown
            print("CLASSIFYING! (Single trigger)")
            
            cv2.imshow("Model Input (What the AI Sees)", canvas)
            
            # Reset everything for the next letter
            points.clear()
            canvas.fill(0)
            prev_x, prev_y = 0, 0
        
        # State 3: "Standby Mode"
        elif current_gesture == "standby":
            points.append(None) # "Lift the pen"
            prev_x, prev_y = 0, 0
        
        # Update the last_gesture at the end of the frame
        last_gesture = current_gesture

    # --- 6. No Hand Found Logic ---
    else:
        # If no hand is found, "lift the pen" and reset state
        points.append(None)
        prev_x, prev_y = 0, 0
        last_gesture = "standby"

    # --- 7. Drawing Logic ---
    # This loop draws the trail based on the 'points' deque
    for i in range(1, len(points)):
        # If either point is None (a "pen up"), skip drawing a line
        if points[i - 1] is None or points[i] is None:
            continue
        
        # Draw the black line on the frame (for the user)
        cv2.line(frame, points[i - 1], points[i], (0, 0, 0), 15)
        
        # Draw a white, thick line on the canvas (for the model)
        cv2.line(canvas, points[i - 1], points[i], (255, 255, 255), 20)

    # --- 8. Display Windows ---
    # Show the main view (webcam + user's drawing)
    cv2.imshow("Air-Writer v0.5 (User View)", frame)

    # --- 9. User Controls ---
    key = cv2.waitKey(1) & 0xFF  # Wait 1ms for a key press
    if key == ord('q'):
        # Quit the application
        break
    elif key == ord('c'):
        # Manual clear
        points.clear()
        canvas.fill(0)
        prev_x, prev_y = 0, 0
        last_gesture = "standby" # Reset state on manual clear

# --- 10. Cleanup ---
cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
hands.close()  # Close the MediaPipe hands solution