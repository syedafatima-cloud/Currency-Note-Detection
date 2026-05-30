import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from preprocess import preprocess_image, calculate_damage_percentage

st.set_page_config(page_title="Currency Note Damage Detection", layout="wide")
st.title("Currency Note Damage and Suspicious Note Detection")
st.markdown("Covers DIP concepts from **Weeks 5–13** of the syllabus.")

uploaded_file = st.file_uploader(
    "Upload Currency Note Image",
    type=["jpg", "png", "jpeg", "jfif", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image.save("temp.jpg")

    results = preprocess_image("temp.jpg")
    damage = calculate_damage_percentage(results["edges"])

    # Damage Classification
    if damage < 5:
        status = "Normal Note"
    elif damage < 10:
        status = "Slightly Damaged"
    elif damage < 20:
        status = "Moderately Damaged"
    else:
        status = "Highly Damaged"

    # ── Final Analysis (shown at the top for quick read) ─────────────────────
    st.subheader("Final Analysis")
    col1, col2, col3 = st.columns(3)
    col1.success(f"Damage Severity: {damage}%")
    col2.info(f"Classification: {status}")
    col3.warning(f"Contours Detected: {results['contour_count']}")

    st.divider()

    # ── Week 2: Original Image ────────────────────────────────────────────────
    st.subheader(" Image Acquisition & Representation")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Original (RGB)")
        st.image(results["original"], channels="BGR")
    with col2:
        st.caption("Grayscale")
        st.image(results["gray"], clamp=True)

    st.divider()

    # ── Week 5: Histogram Processing ─────────────────────────────────────────
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

    # ── Week 6: Spatial Filtering ─────────────────────────────────────────────
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

    # ── Week 7: Edge Detection ────────────────────────────────────────────────
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

    # ── Week 8–9: Frequency Domain (FFT) ─────────────────────────────────────
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

    # ── Week 10: Noise Model & Restoration ───────────────────────────────────
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

    # ── Week 11: Segmentation — Contours & Hough Transform ───────────────────
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

    # ── Week 12: Thresholding ─────────────────────────────────────────────────
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

    # ── Week 13: Morphological Processing ────────────────────────────────────
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