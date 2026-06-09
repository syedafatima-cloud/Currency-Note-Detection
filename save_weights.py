"""Run once locally to extract weights from the full model file.

    python save_weights.py

Produces fake_detector_weights.h5 which can be loaded by any TF 2.x version.
"""
from tensorflow.keras.models import load_model

model = load_model("fake_detector.h5")
model.save_weights("fake_detector_weights.weights.h5")
print("Saved -> fake_detector_weights.weights.h5")
