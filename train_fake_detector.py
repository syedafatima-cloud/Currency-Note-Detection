"""
Trains a binary MobileNetV2 classifier to detect real vs fake Pakistani currency.

Prerequisites:
    python download_fake_dataset.py   # organises dataset/fake_detection/real & fake

Outputs:
    fake_detector.h5   — saved model
    fake_detector_labels.json — {"fake": 0, "real": 1} (or whatever flow_from_directory assigns)
"""

import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATASET_PATH = "dataset/Real And  Fake Currency Dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 25

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    shear_range=0.05,
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
)

val_gen = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
)

print(f"Classes: {train_gen.class_indices}")
print(f"Training samples  : {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}\n")

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks_p1 = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ModelCheckpoint("fake_detector_best.h5", save_best_only=True, verbose=1),
]

print("=" * 60)
print("Phase 1: Training classifier head (base frozen)")
print("=" * 60)
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_PHASE1, callbacks=callbacks_p1)

# ------------------------------------------------------------------
# Phase 2 — fine-tune last 30 base layers
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Phase 2: Fine-tuning last 30 base layers")
print("=" * 60)
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks_p2 = [
    EarlyStopping(patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ModelCheckpoint("fake_detector_best.h5", save_best_only=True, verbose=1),
]

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_PHASE2, callbacks=callbacks_p2)

# ------------------------------------------------------------------
# Evaluate and save
# ------------------------------------------------------------------
loss, accuracy = model.evaluate(val_gen, verbose=1)
print(f"\nFinal Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Final Validation Loss    : {loss:.4f}")

model.save("fake_detector.h5")
print("\nSaved model  -> fake_detector.h5")

with open("fake_detector_labels.json", "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)
print("Saved labels -> fake_detector_labels.json")
