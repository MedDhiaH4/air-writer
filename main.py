# main.py

# --- 1. Imports ---
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from preprocessing import preprocess_image  # Import our processing function
from data import CLASS_NAMES              # Import our class names list

# --- 2. Global Setup & Constants ---

points = deque(maxlen=128)
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Initialize MediaPipe
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

# Smoothing variables
prev_x, prev_y = 0, 0
smooth_factor = 0.5

# Gesture state
last_gesture = "standby"

# --- 3. Load Model and Set Up Text ---
print("Loading trained model: air_writer_model.h5...")
model = tf.keras.models.load_model('air_writer_model.h5')
print("Model loaded successfully.")

# This list will hold our written letters
sentence = []
text_font = cv2.FONT_HERSHEY_SIMPLEX
text_scale = 2
text_thickness = 3

# --- 4. Webcam Initialization ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("Starting Air-Writer v1.0 (Live Mode)...")
print("- Show ONE (index) finger to write.")
print("- Show FIVE (full palm) fingers to classify & clear.")
print("- Show ONE (pinky) finger to delete/clear.")
print("- Press 'c' to clear manually. Press 'q' to quit.")

# --- 5. Main Application Loop ---
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # --- 6. Hand Landmark & Gesture Logic ---
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
        current_gesture = "standby"
        if fingers_up[1] and not fingers_up[0] and not fingers_up[2] and not fingers_up[3] and not fingers_up[4]:
            current_gesture = "write"
        elif all(fingers_up):
            current_gesture = "classify"
        elif fingers_up[4] and not fingers_up[0] and not fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
            current_gesture = "delete"
        
        # --- State Actions ---
        
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
        
        # State 2: "Classify Mode" (Triggers model prediction)
        elif current_gesture == "classify" and last_gesture != "classify":
            print("CLASSIFYING!")
            
            # Preprocess the canvas to get the 28x28 letter
            processed_img = preprocess_image(canvas, size=28)
            
            # Check if the canvas was blank (is 'space')
            if np.all(processed_img == 0):
                print("  ...Result: SPACE")
                sentence.append(" ")
            else:
                # Canvas has a letter, predict it
                # 1. Reshape for the model: (1, 28, 28, 1)
                img_for_model = np.expand_dims(processed_img, axis=0)
                img_for_model = np.expand_dims(img_for_model, axis=-1)
                
                # 2. Normalize the image (THE CRITICAL FIX)
                # We cast to float32 AND divide by 255.0 to match the training data
                img_for_model = tf.cast(img_for_model, tf.float32) / 255.0

                # 3. Predict
                prediction = model.predict(img_for_model)
                predicted_index = np.argmax(prediction)
                predicted_char = CLASS_NAMES[predicted_index]
                
                print(f"  ...Result: {predicted_char}")
                sentence.append(predicted_char)
            
            # Reset everything for the next letter
            points.clear()
            canvas.fill(0)
            prev_x, prev_y = 0, 0
        
        # State 3: "Delete Mode" (Removes last letter)
        elif current_gesture == "delete" and last_gesture != "delete":
            print("DELETE!")
            
            if len(sentence) > 0:
                sentence.pop() # Remove the last letter from our list
            
            # Also clear the current drawing
            points.clear()
            canvas.fill(0)
            prev_x, prev_y = 0, 0

        # State 4: "Standby Mode"
        elif current_gesture == "standby":
            points.append(None) # "Lift the pen"
            prev_x, prev_y = 0, 0
        
        # Update the last_gesture at the end of the frame
        last_gesture = current_gesture

    # --- 7. No Hand Found Logic ---
    else:
        points.append(None)
        prev_x, prev_y = 0, 0
        last_gesture = "standby"

    # --- 8. Drawing Logic ---
    # Draw the user's trail
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        
        cv2.line(frame, points[i - 1], points[i], (0, 0, 0), 15)
        
        # THE SECOND CRITICAL FIX: Use the thickness you found
        cv2.line(canvas, points[i - 1], points[i], (255, 255, 255), 20)

    # --- 9. Display Written Text ---
    # Join the list of letters into a single string
    text_to_display = "".join(sentence)
    
    if text_to_display:
        # Get the size of the text to draw a background rectangle
        (w, h), baseline = cv2.getTextSize(text_to_display, text_font, text_scale, text_thickness)
        
        # Y-coordinate for the text baseline
        text_y_pos = 60
        
        # Draw the black rectangle background
        cv2.rectangle(frame, 
                      (10, text_y_pos - h - baseline - 10), 
                      (10 + w + 10, text_y_pos + baseline + 5), 
                      (0, 0, 0), 
                      cv2.FILLED)
        
        # Draw the white text
        cv2.putText(frame, 
                    text_to_display, 
                    (15, text_y_pos), 
                    text_font, 
                    text_scale, 
                    (255, 255, 255), 
                    text_thickness)

    # --- 10. Display Window ---
    cv2.imshow("Air-Writer v1.0 (Live Mode)", frame)

    # --- 11. User Controls ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        points.clear()
        canvas.fill(0)
        prev_x, prev_y = 0, 0
        last_gesture = "standby"

# --- 12. Cleanup ---
cap.release()
cv2.destroyAllWindows()
hands.close()

