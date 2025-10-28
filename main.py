import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

from preprocessing import preprocess_image # Handles cropping/resizing
from data import CLASS_NAMES              # List of '0'-'9', 'A'-'Z'

# --- Global Setup & Constants ---
points = deque(maxlen=128) # Stores the visible trail points
# Canvas for the model's input (white drawing on black background)
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7,
                       max_num_hands=1)

# Landmark indices for finger detection
finger_tip_ids = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                  mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                  mp_hands.HandLandmark.PINKY_TIP]
finger_pip_ids = [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
                  mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP,
                  mp_hands.HandLandmark.PINKY_PIP]

# Smoothing variables for the visible trail
prev_x, prev_y = 0, 0
smooth_factor = 0.5

# Gesture state tracking
last_gesture = "standby"

# --- Load Model & Setup Text ---
print("Loading trained model: air_writer_model.h5...") # Using the non-final model name from user paste
model = tf.keras.models.load_model('air_writer_model.h5')
print("Model loaded successfully.")

sentence = [] # Stores the recognized characters
text_font = cv2.FONT_HERSHEY_SIMPLEX
text_scale = 2
text_thickness = 3

# --- Webcam Initialization ---
cap = cv2.VideoCapture(0) # Use camera index 0
cap.set(3, 1280) # Set desired width
cap.set(4, 720)  # Set desired height

print("Starting Air-Writer...")
print("- Index finger: Write")
print("- Full palm: Classify & Clear")
print("- Pinky finger: Delete")
print("- Press 'c': Manual Clear | Press 'q': Quit")

# --- Main Application Loop ---
while True:
    success, frame = cap.read()
    if not success:
        print("ERROR: Failed to read frame from camera.")
        break

    frame = cv2.flip(frame, 1) # Mirror view

    # Process frame for hand detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_gesture = "standby" # Default gesture for this frame

    # --- Hand Landmark & Gesture Logic ---
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark

        # --- Finger Counting ---
        fingers_up = [False] * 5
        # Check thumb orientation (uses x-axis relative to pip)
        if lm[finger_tip_ids[0]].x < lm[finger_pip_ids[0]].x: fingers_up[0] = True
        # Check other fingers' vertical extension (y-axis relative to pip)
        for i in range(1, 5):
            if lm[finger_tip_ids[i]].y < lm[finger_pip_ids[i]].y: fingers_up[i] = True

        # --- State Machine ---
        # Determine gesture based on fingers up
        if fingers_up[1] and not any(fingers_up[i] for i in [0, 2, 3, 4]):
            current_gesture = "write"
        elif all(fingers_up):
            current_gesture = "classify"
        elif fingers_up[4] and not any(fingers_up[i] for i in [0, 1, 2, 3]):
            current_gesture = "delete"

        # --- State Actions ---
        if current_gesture == "write":
            h, w, c = frame.shape
            # Get index finger coordinates
            cx = int(lm[finger_tip_ids[1]].x * w)
            cy = int(lm[finger_tip_ids[1]].y * h)

            # Initialize smoother on first point
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = cx, cy

            # Apply smoothing to visible trail point
            smooth_cx = int(prev_x * (1 - smooth_factor) + cx * smooth_factor)
            smooth_cy = int(prev_y * (1 - smooth_factor) + cy * smooth_factor)

            points.append((smooth_cx, smooth_cy)) # Add smoothed point to visible trail
            prev_x, prev_y = smooth_cx, smooth_cy # Update previous point for smoothing

            # Draw on internal canvas using non-smoothed points for sharper input?
            # Let's keep using smoothed points for consistency for now.
            if len(points) > 1 and points[-2] is not None:
                cv2.line(canvas, points[-2], points[-1], (255, 255, 255), 20) # White, thick line

            # Draw feedback circle on displayed frame
            cv2.circle(frame, (smooth_cx, smooth_cy), 10, (0, 0, 0), cv2.FILLED)

        elif current_gesture == "classify" and last_gesture != "classify":
            print("CLASSIFYING!")
            processed_img = preprocess_image(canvas, size=28) # Preprocess internal canvas

            if np.all(processed_img == 0): # Check if blank
                print("  ...Result: SPACE")
                sentence.append(" ")
            else: # Predict character
                img_for_model = np.expand_dims(processed_img, axis=(0, -1)) # Reshape
                img_for_model = tf.cast(img_for_model, tf.float32) / 255.0 # Normalize
                try:
                    prediction = model.predict(img_for_model)
                    predicted_index = np.argmax(prediction)
                    predicted_char = CLASS_NAMES[predicted_index]
                    print(f"  ...Result: {predicted_char}")
                    sentence.append(predicted_char)
                except Exception as e:
                     print(f"ERROR during prediction: {e}")
                     sentence.append("#") # Error character

            # Reset state
            points.clear()
            canvas.fill(0)
            prev_x, prev_y = 0, 0

        elif current_gesture == "delete" and last_gesture != "delete":
            print("DELETE!")
            if len(sentence) > 0: sentence.pop() # Remove last character
            # Reset state
            points.clear()
            canvas.fill(0)
            prev_x, prev_y = 0, 0

        elif current_gesture == "standby":
            points.append(None) # Lift the pen for visible trail
            prev_x, prev_y = 0, 0 # Reset smoother

    else: # No hand detected
        points.append(None)
        prev_x, prev_y = 0, 0
        current_gesture = "standby"

    # Update last gesture for next frame's comparison
    last_gesture = current_gesture

    # --- Drawing Logic on Display Frame ---
    # Draw the visible trail
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(frame, points[i - 1], points[i], (0, 0, 0), 15) # Black line for user

    # --- Display Written Text Overlay ---
    text_to_display = "".join(sentence)
    if text_to_display:
        (w, h), baseline = cv2.getTextSize(text_to_display, text_font, text_scale, text_thickness)
        text_y_pos = 60 # Position near top-left
        # Draw background rectangle
        cv2.rectangle(frame, (10, text_y_pos - h - baseline - 10), (10 + w + 10, text_y_pos + baseline + 5), (0, 0, 0), cv2.FILLED)
        # Draw text
        cv2.putText(frame, text_to_display, (15, text_y_pos), text_font, text_scale, (255, 255, 255), text_thickness)

    # --- Display Window ---
    cv2.imshow("Air-Writer", frame) # Simplified title

    # --- User Controls ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("'q' pressed, exiting.")
        break
    elif key == ord('c'):
        print("Clearing drawing and sentence.")
        points.clear()
        canvas.fill(0)
        prev_x, prev_y = 0, 0
        last_gesture = "standby"
        sentence = [] # Also clear sentence list

# --- Cleanup ---
print("Releasing camera and closing windows...")
cap.release()
cv2.destroyAllWindows()
# Check if 'hands' object exists and has 'close' method before calling
if 'hands' in locals() and hasattr(hands, 'close'):
    hands.close()
print("Application finished.")