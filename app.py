import io
import json
import os
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from preprocess import preprocess_from_array, calculate_damage_percentage, detect_note_presence

st.set_page_config(page_title="Currency Note Damage Detection", layout="wide")
st.title("Currency Note Damage and Suspicious Note Detection")

# ------------------------------------------------------------------
# Model loaders — run once for the whole session
# ------------------------------------------------------------------
@st.cache_resource
def load_currency_model():
    if not os.path.exists("currency_model.h5") or not os.path.exists("class_labels.json"):
        return None, None
    mdl = load_model("currency_model.h5")
    with open("class_labels.json") as f:
        idx_to_label = {v: k for k, v in json.load(f).items()}
    return mdl, idx_to_label


@st.cache_resource
def load_fake_detector():
    if not os.path.exists("fake_detector.h5") or not os.path.exists("fake_detector_labels.json"):
        return None, None
    mdl = load_model("fake_detector.h5")
    with open("fake_detector_labels.json") as f:
        idx_to_class = {v: k for k, v in json.load(f).items()}
    return mdl, idx_to_class


currency_model, idx_to_label = load_currency_model()
fake_detector, fake_idx_to_class = load_fake_detector()

# ------------------------------------------------------------------
# Per-image processing — cached by image bytes so the same image
# is never reprocessed on a Streamlit re-render
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Analysing image…")
def run_preprocess(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    results = preprocess_from_array(bgr)
    damage = calculate_damage_percentage(results["edges"])
    return results, damage


@st.cache_data(show_spinner=False)
def run_denomination(file_bytes: bytes, _model, _idx_to_label):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((224, 224))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = _model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    label = _idx_to_label[top_idx]
    confidence = float(probs[top_idx]) * 100
    parts = label.split("_")
    denomination = parts[0]
    side = parts[1].capitalize() if len(parts) > 1 else ""
    return f"Rs {denomination} ({side})", confidence


@st.cache_data(show_spinner=False)
def run_authenticity(file_bytes: bytes, _model, _idx_to_class):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((224, 224))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = float(_model.predict(arr, verbose=0)[0][0])
    fake_class_idx = next(
        (k for k, v in _idx_to_class.items() if "fake" in v.lower()), None
    )
    fake_prob = prob if fake_class_idx == 1 else 1.0 - prob
    real_prob = 1.0 - fake_prob
    confidence = max(fake_prob, real_prob) * 100
    if confidence < 70:
        label = "Uncertain"
    elif fake_prob >= 0.5:
        label = "Suspicious / Fake"
    else:
        label = "Genuine"
    return label, confidence, fake_prob * 100


# ------------------------------------------------------------------
# Upload
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Currency Note Image",
    type=["jpg", "png", "jpeg", "jfif", "webp"]
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    results, damage = run_preprocess(file_bytes)

    # ── Note presence check — reject non-note images early ───────────────────
    note_detected = detect_note_presence(results)

    if not note_detected:
        st.error("No currency note detected in this image. Please upload a clear photo of a Pakistani currency note.")
    else:
        if damage < 5:
            status = "Normal Note"
        elif damage < 10:
            status = "Slightly Damaged"
        elif damage < 20:
            status = "Moderately Damaged"
        else:
            status = "Highly Damaged"

        # ── Suspicious / Fake Note Detection ─────────────────────────────────
        if fake_detector is not None:
            auth_label, auth_conf, fake_pct = run_authenticity(
                file_bytes, fake_detector, fake_idx_to_class
            )
            st.subheader("Suspicious Note Detection")
            acol1, acol2, acol3 = st.columns(3)
            if auth_label == "Genuine":
                acol1.success(f"Result: {auth_label}")
            elif auth_label == "Uncertain":
                acol1.warning("Result: Uncertain — please upload a clear currency note image")
            else:
                acol1.error(f"Result: {auth_label}")
            acol2.info(f"Confidence: {auth_conf:.1f}%")
            acol3.metric("Fake probability", f"{fake_pct:.1f}%")
            st.divider()
        else:
            st.info("Fake/real detector not loaded yet — run `python train_fake_detector.py` first.")

        # ── Denomination Detection ────────────────────────────────────────────
        if currency_model is not None:
            denomination_label, denom_conf = run_denomination(
                file_bytes, currency_model, idx_to_label
            )
            st.subheader("Denomination Detection")
            dcol1, dcol2 = st.columns(2)
            dcol1.success(f"Detected: {denomination_label}")
            dcol2.info(f"Confidence: {denom_conf:.1f}%")
            st.divider()

        # ── Final Analysis ────────────────────────────────────────────────────
        st.subheader("Final Analysis")
        col1, col2, col3 = st.columns(3)
        col1.success(f"Damage Severity: {damage}%")
        col2.info(f"Classification: {status}")
        col3.warning(f"Contours Detected: {results['contour_count']}")

        st.divider()

        # ── Week 2: Original Image ────────────────────────────────────────────
        st.subheader(" Image Acquisition & Representation")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Original (RGB)")
            st.image(results["original"], channels="BGR")
        with col2:
            st.caption("Grayscale")
            st.image(results["gray"], clamp=True)

        st.divider()

        # ── Week 5: Histogram Processing ─────────────────────────────────────
        st.subheader(" Histogram Processing")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("Before equalization (grayscale)")
            st.image(results["gray"], clamp=True)
        with col2:
            st.caption("After histogram equalization")
            st.image(results["equalized"], clamp=True)
        with col3:
            st.caption("Pixel intensity histogram")
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(results["histogram"], color="steelblue")
            ax.set_title("Intensity Histogram")
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            plt.close(fig)

        st.divider()

        # ── Week 6: Spatial Filtering ─────────────────────────────────────────
        st.subheader(" Spatial Domain Filtering (Gaussian Blur)")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Equalized image (before blur)")
            st.image(results["equalized"], clamp=True)
        with col2:
            st.caption("After Gaussian Blur (5×5 kernel)")
            st.image(results["blurred"], clamp=True)
        st.info(
            "Gaussian blur is a smoothing (low-pass) spatial filter. "
            "It reduces high-frequency noise before edge detection."
        )

        st.divider()

        # ── Week 7: Edge Detection ────────────────────────────────────────────
        st.subheader(" Edge Detection (Canny)")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Blurred image (input)")
            st.image(results["blurred"], clamp=True)
        with col2:
            st.caption("Canny edges (thresholds: 100, 200)")
            st.image(results["edges"], clamp=True)
        st.info(
            "Canny uses 1st-derivative gradients (Sobel) to find intensity changes. "
            "Edges on currency notes reveal tears, folds, and damage."
        )

        st.divider()

        # ── Week 8–9: Frequency Domain (FFT) ─────────────────────────────────
        st.subheader(" Frequency Domain Analysis (FFT)")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Grayscale image (spatial domain)")
            st.image(results["gray"], clamp=True)
        with col2:
            st.caption("FFT magnitude spectrum (frequency domain)")
            st.image(results["fft"], clamp=True)
        st.info(
            "The FFT transforms the image from the spatial domain to the frequency domain. "
            "Bright regions near the centre = low frequencies (smooth areas). "
            "Bright regions at the edges = high frequencies (fine details, edges, noise). "
            "This reveals the frequency content of the currency note."
        )

        st.divider()

        # ── Week 10: Noise Model & Restoration ───────────────────────────────
        st.subheader(" Noise Model & Restoration")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("Original grayscale")
            st.image(results["gray"], clamp=True)
        with col2:
            st.caption("Salt & pepper noise added (2% each)")
            st.image(results["noisy"], clamp=True)
        with col3:
            st.caption("Noise removed — Median Filter (5×5)")
            st.image(results["denoised"], clamp=True)
        st.info(
            "Salt & pepper noise randomly sets pixels to 0 (pepper) or 255 (salt). "
            "The median filter replaces each pixel with the median of its neighbourhood, "
            "which effectively removes impulse noise while preserving edges."
        )

        st.divider()

        # ── Week 11: Segmentation — Contours & Hough Transform ───────────────
        st.subheader(" Segmentation: Contours & Hough Line Transform")
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"Contour detection ({results['contour_count']} contours found)")
            st.image(results["contours"], channels="BGR")
        with col2:
            st.caption("Hough Line Transform (blue lines)")
            st.image(results["hough"], channels="BGR")
        st.info(
            "Contours trace the boundaries of connected regions in the edge image. "
            "The Hough transform detects straight lines — useful for detecting "
            "the note's borders and printed grid patterns."
        )

        st.divider()

        # ── Week 12: Thresholding ─────────────────────────────────────────────
        st.subheader("Thresholding: Global vs Adaptive")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Global thresholding (fixed threshold = 127)")
            st.image(results["threshold"], clamp=True)
        with col2:
            st.caption("Adaptive thresholding (Gaussian, block=11, C=2)")
            st.image(results["adaptive_thresh"], clamp=True)
        st.info(
            "Global thresholding applies a single fixed value to the whole image — "
            "it fails when lighting is uneven. "
            "Adaptive thresholding computes a local threshold for each region, "
            "producing much better results on currency notes with varying illumination."
        )

        st.divider()

        # ── Week 13: Morphological Processing ────────────────────────────────
        st.subheader(" Morphological Operations")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Morphological Closing (fills small holes)")
            st.image(results["morphology"], clamp=True)
        with col2:
            st.caption("Morphological Opening (removes small blobs)")
            st.image(results["morphology_open"], clamp=True)
        st.info(
            "Closing = Dilation followed by Erosion. It joins broken contours and fills gaps. "
            "Opening = Erosion followed by Dilation. It removes small noise specks. "
            "Both use a 5×5 structuring element."
        )
