import cv2
import numpy as np

IMG_SIZE = 600


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return _process(image)


def preprocess_from_array(bgr_image):
    """Same as preprocess_image but accepts a BGR numpy array directly."""
    image = cv2.resize(bgr_image, (IMG_SIZE, IMG_SIZE))
    return _process(image)


def _process(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ── Week 5: Histogram Equalization ──────────────────────────────────────
    equalized = cv2.equalizeHist(gray)

    # ── Week 6: Gaussian Blur (spatial smoothing) ────────────────────────────
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

    # ── Week 7: Edge Detection (Canny) ───────────────────────────────────────
    edges = cv2.Canny(blurred, 100, 200)

    # ── Week 10: Noise Model — Salt & Pepper ─────────────────────────────────
    noisy = add_salt_pepper_noise(gray, salt_prob=0.02, pepper_prob=0.02)

    # ── Week 10: Noise Removal — Median Filter ───────────────────────────────
    denoised = cv2.medianBlur(noisy, 5)

    # ── Week 12: Global Thresholding ─────────────────────────────────────────
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # ── Week 12: Adaptive Thresholding ───────────────────────────────────────
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # ── Week 13: Morphological Operations ────────────────────────────────────
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Also apply opening to remove small noise blobs
    morph_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # ── Week 11: Contour Detection ───────────────────────────────────────────
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # ── Week 8–9: Frequency Domain — FFT Magnitude Spectrum ──────────────────
    fft_image, magnitude_spectrum = compute_fft(gray)

    # ── Week 11: Hough Line Transform ────────────────────────────────────────
    hough_image = detect_hough_lines(image.copy(), edges)

    # ── Week 5: Pixel Intensity Histogram ────────────────────────────────────
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])

    return {
        "original":           image,
        "gray":               gray,
        "equalized":          equalized,
        "blurred":            blurred,
        "noisy":              noisy,
        "denoised":           denoised,
        "edges":              edges,
        "threshold":          thresh,
        "adaptive_thresh":    adaptive_thresh,
        "morphology":         morph,
        "morphology_open":    morph_open,
        "contours":           contour_image,
        "fft":                fft_image,
        "magnitude_spectrum": magnitude_spectrum,
        "hough":              hough_image,
        "histogram":          histogram,
        "contour_count":      len(contours),
    }


# ── Week 10: Salt & Pepper Noise Model ───────────────────────────────────────
def add_salt_pepper_noise(gray_image, salt_prob=0.02, pepper_prob=0.02):
    noisy = gray_image.copy()
    total_pixels = gray_image.size

    # Salt (white) noise
    num_salt = int(total_pixels * salt_prob)
    salt_coords = (
        np.random.randint(0, gray_image.shape[0], num_salt),
        np.random.randint(0, gray_image.shape[1], num_salt),
    )
    noisy[salt_coords] = 255

    # Pepper (black) noise
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = (
        np.random.randint(0, gray_image.shape[0], num_pepper),
        np.random.randint(0, gray_image.shape[1], num_pepper),
    )
    noisy[pepper_coords] = 0

    return noisy


# ── Week 8–9: FFT — Frequency Domain Analysis ────────────────────────────────
def compute_fft(gray_image):
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # Normalise to 0–255 for display
    mag_norm = cv2.normalize(
        magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    return mag_norm, magnitude_spectrum


# ── Week 11: Hough Line Transform ────────────────────────────────────────────
def detect_hough_lines(image, edges):
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=10,
    )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image


# ── Damage Severity Calculation ───────────────────────────────────────────────
def calculate_damage_percentage(edges):
    damaged_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    damage_percentage = (damaged_pixels / total_pixels) * 100
    return round(damage_percentage, 2)


# ── Fake Note Detection — Template Matching ───────────────────────────────────
def template_match(test_image, template_image):
    result = cv2.matchTemplate(
        test_image,
        template_image,
        cv2.TM_CCOEFF_NORMED
    )
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val