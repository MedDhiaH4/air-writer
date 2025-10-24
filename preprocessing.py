import cv2
import numpy as np

def preprocess_image(canvas, size=64):
    """
    Takes the full-size canvas, finds the bounding box of the drawing,
    crops it, pads it to a square, and resizes it to a standard size.
    """
    
    # 1. Convert to Grayscale
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # 2. Find Contours
    # RETR_EXTERNAL gets only the outermost contours
    contours, _ = cv2.findContours(gray_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Safety Check
    if not contours:
        # Return a blank 64x64 image (will be treated as 'space')
        return np.zeros((size, size), dtype=np.uint8)

    # 4. Get Bounding Box for *all* contours
    all_contours = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_contours)

    # 5. Crop the image to the bounding box
    cropped_image = gray_canvas[y:y+h, x:x+w]

    # 6. Pad to Square (to maintain aspect ratio)
    max_dim = max(w, h)
    square_image = np.zeros((max_dim, max_dim), dtype=np.uint8)

    # Calculate offsets to center the image
    offset_x = (max_dim - w) // 2
    offset_y = (max_dim - h) // 2

    # Paste the cropped image into the center of the square
    square_image[offset_y:offset_y+h, offset_x:offset_x+w] = cropped_image

    # 7. Resize to the final standard size (e.g., 64x64)
    # INTER_AREA is good for shrinking images
    final_image = cv2.resize(square_image, (size, size), interpolation=cv2.INTER_AREA)

    return final_image