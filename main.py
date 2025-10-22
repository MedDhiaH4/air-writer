# main.py
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# --- Global Setup ---

# This will store the points in our drawing
points = deque(maxlen=128)

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Start the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Set width
cap.set(4, 720)  # Set height

print("Starting Air-Writer... Press 'c' to clear screen. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally (like a selfie)
    frame = cv2.flip(frame, 1)

    # Create a black canvas to draw on, same size as the frame
    canvas = np.zeros_like(frame)

    # Convert the BGR frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find hands
    results = hands.process(frame_rgb)

    # --- Hand Tracking Logic ---
    if results.multi_hand_landmarks:
        # We only set max_num_hands=1, so we just take the first hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get the (x, y) coordinate of the index finger tip (landmark #8)
        h, w, c = frame.shape
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

        # Add the point to our deque
        points.append((cx, cy))

        # Draw a small circle on the fingertip
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    # --- Drawing Logic ---
    # Draw the trail on the black canvas
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        # Draw a line between consecutive points
        cv2.line(canvas, points[i - 1], points[i], (255, 0, 0), 5) # Blue trail

    # Combine the frame and the canvas
    frame = cv2.addWeighted(frame, 1.0, canvas, 0.8, 0)

    # Show the final frame
    cv2.imshow("Air-Writer v0.1", frame)

    # --- Controls ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Quit
        break
    elif key == ord('c'):
        # Clear the drawing
        points.clear()

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
hands.close()