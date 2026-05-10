import cv2
import numpy as np

IMG_SIZE = 128


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Noise reduction
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Histogram Equalization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # Edge Detection
    edges = cv2.Canny(gray, 100, 200)

    # Convert back to 3 channels
    processed = cv2.merge([gray, gray, gray])

    processed = processed / 255.0

    return processed