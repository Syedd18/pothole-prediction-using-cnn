import os
import glob
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------
# Settings
size = 100
epochs = 50
batch_size = 32
dataset_path = r"C:\Users\acer\Downloads\pothole-detection-system-using-convolution-neural-networks-master\pothole-detection-system-using-convolution-neural-networks-master\My Dataset"

# ------------------------
# Supported image formats
extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']

# ------------------------
# Load images function
def load_images(folder):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    images = []
    for path in files:
        img = cv2.imread(path, 0)  # grayscale
        if img is None:
            print(f"Warning: {path} cannot be read, skipping.")
            continue
        img = cv2.resize(img, (size, size))
        images.append(img)
    return np.asarray(images, dtype=np.float32) / 255.0

# ------------------------
# Load training data
X_pothole = load_images(os.path.join(dataset_path, 'train/Pothole'))
X_plain = load_images(os.path.join(dataset_path, 'train/Plain'))

X_train = np.concatenate((X_pothole, X_plain), axis=0)
y_train = np.concatenate((np.ones(len(X_pothole)), np.zeros(len(X_plain))), axis=0)

# Shuffle
X_train, y_train = shuffle(X_train, y_train)

# Reshape for CNN
X_train = X_train.reshape(-1, size, size, 1)
y_train = to_categorical(y_train)

# ------------------------
# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.1
)

# ------------------------
# Build CNN
model = Sequential()
model.add(Conv2D(32, (5,5), activation='relu', padding='same', input_shape=(size,size,1)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------
# Train model
model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size, subset='training'),
    validation_data=datagen.flow(X_train, y_train, batch_size=batch_size, subset='validation'),
    epochs=epochs
)

# ------------------------
# Save model
model.save(os.path.join(os.path.dirname(dataset_path), 'sample_manual_augmented.h5'))
print("Model saved successfully!")
