import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATASET_PATH = "dataset/data-rescaled"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 30

# ------------------------------------------------------------------
# Data generators with augmentation
# ------------------------------------------------------------------
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
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_gen = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

NUM_CLASSES = len(train_gen.class_indices)
print(f"\nDetected {NUM_CLASSES} classes:")
for cls, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
    print(f"  [{idx}] {cls}")
print(f"\nTraining samples : {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}\n")

# ------------------------------------------------------------------
# Model — MobileNetV2 transfer learning
# ------------------------------------------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_phase1 = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ModelCheckpoint("currency_model_best.h5", save_best_only=True, verbose=1),
]

# ------------------------------------------------------------------
# Phase 1 — train classifier head only
# ------------------------------------------------------------------
print("=" * 60)
print("Phase 1: Training classifier head (base frozen)")
print("=" * 60)
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks_phase1,
)

# ------------------------------------------------------------------
# Phase 2 — fine-tune last 30 layers of base
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Phase 2: Fine-tuning last 30 base layers")
print("=" * 60)
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_phase2 = [
    EarlyStopping(patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ModelCheckpoint("currency_model_best.h5", save_best_only=True, verbose=1),
]

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks_phase2,
)

# ------------------------------------------------------------------
# Evaluate and save
# ------------------------------------------------------------------
loss, accuracy = model.evaluate(val_gen, verbose=1)
print(f"\nFinal Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Final Validation Loss    : {loss:.4f}")

model.save("currency_model.h5")
print("\nSaved model  → currency_model.h5")

with open("class_labels.json", "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)
print("Saved labels → class_labels.json")
