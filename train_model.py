import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from preprocess import preprocess_image

DATASET_PATH = "dataset"

CATEGORIES = [
    "normal",
    "torn",
    "stained",
    "faded",
    "fake"
]

X = []
y = []

for category in CATEGORIES:
    folder = os.path.join(DATASET_PATH, category)

    label = CATEGORIES.index(category)

    for img_name in os.listdir(folder):
        try:
            path = os.path.join(folder, img_name)

            image = preprocess_image(path)

            X.append(image)
            y.append(label)

        except:
            pass

X = np.array(X)
y = np.array(y)

y = to_categorical(y, num_classes=len(CATEGORIES))
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(CATEGORIES), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Accuracy: {accuracy * 100:.2f}%")

model.save("model.h5")

print("Model Saved Successfully")