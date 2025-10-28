import cv2
import numpy as np

def preprocess_image(canvas, size=28):
    """
    Finds drawing bbox, crops, pads to square, resizes to target size.
    Returns a blank image if canvas is empty.
    """
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # Find contours of the drawing
    contours, _ = cv2.findContours(gray_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros((size, size), dtype=np.uint8) # Return blank for space

    # Get single bounding box around all contours
    all_contours = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_contours)

    # Crop to bounding box
    cropped_image = gray_canvas[y:y+h, x:x+w]

    # Pad to square to maintain aspect ratio
    max_dim = max(w, h)
    square_image = np.zeros((max_dim, max_dim), dtype=np.uint8)
    offset_x = (max_dim - w) // 2
    offset_y = (max_dim - h) // 2
    square_image[offset_y:offset_y+h, offset_x:offset_x+w] = cropped_image

    # Resize to final size (e.g., 28x28)
    final_image = cv2.resize(square_image, (size, size), interpolation=cv2.INTER_AREA)

    return final_image