import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# ----------------------------
# CONFIG
# ----------------------------

IMAGE_SIZE = 128
DATASET_PATH = "dataset"
EPOCHS = 20
BATCH_SIZE = 16

# ----------------------------
# LOAD DATA
# ----------------------------

data = []
labels = []

tumor_path = os.path.join(DATASET_PATH, "yes")
no_tumor_path = os.path.join(DATASET_PATH, "no")

def load_images(folder, label):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img = np.array(img)

            data.append(img)
            labels.append(label)

        except:
            continue


print("Loading dataset...")

load_images(tumor_path, 1)
load_images(no_tumor_path, 0)

data = np.array(data)
labels = np.array(labels)

print("Total images loaded:", len(data))

# ----------------------------
# PREPROCESSING
# ----------------------------

data = data / 255.0
labels = to_categorical(labels, 2)

x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

# ----------------------------
# DATA AUGMENTATION
# ----------------------------

datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

datagen.fit(x_train)

# ----------------------------
# CNN MODEL
# ----------------------------

model = Sequential()

model.add(Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(2,2))

model.add(BatchNormalization())

model.add(Conv2D(128, (3,3), activation="relu"))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(2, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# EARLY STOPPING
# ----------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

# ----------------------------
# TRAIN MODEL
# ----------------------------

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    callbacks=[early_stop]
)

# ----------------------------
# EVALUATION
# ----------------------------

print("\nEvaluating model...")

loss, accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", accuracy)

# classification metrics
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# ----------------------------
# SAVE MODEL
# ----------------------------

model.save("brain_tumor_cnn_model.h5")

print("Model saved successfully.")

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------

def predict_image(image_path):

    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img)

    img = img / 255.0
    img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)

    prediction = model.predict(img)

    if np.argmax(prediction) == 1:
        print("Tumor Detected")
    else:
        print("No Tumor Detected")


# Example usage
# predict_image("sample_mri.jpg")