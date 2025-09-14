import numpy as np
import cv2
import glob
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

# ------------------------
# Global settings
# ------------------------
size = 100
inputShape = (size, size, 1)

# ------------------------
# Safe image loader
# ------------------------
def load_images(file_paths):
    images = []
    for img_path in file_paths:
        img = cv2.imread(img_path, 0)
        if img is None:
            print(f"Warning: Cannot read file {img_path}, skipping.")
            continue
        img = cv2.resize(img, (size, size))
        images.append(img)
    return np.asarray(images, dtype=np.float32) / 255.0  # normalize

# ------------------------
# Automatically load all training images
# ------------------------
pothole_train_files = glob.glob("C:/Users/acer/Downloads/My Dataset/train/Pothole/*.jpg") + \
                      glob.glob("C:/Users/acer/Downloads/My Dataset/train/Pothole/*.jpeg") + \
                      glob.glob("C:/Users/acer/Downloads/My Dataset/train/Pothole/*.png")

plain_train_files = glob.glob("C:/Users/acer/Downloads/My Dataset/train/Plain/*.jpg") + \
                    glob.glob("C:/Users/acer/Downloads/My Dataset/train/Plain/*.jpeg") + \
                    glob.glob("C:/Users/acer/Downloads/My Dataset/train/Plain/*.png")

train_pothole = load_images(pothole_train_files)
train_plain = load_images(plain_train_files)

# ------------------------
# Automatically load all testing images
# ------------------------
pothole_test_files = glob.glob("C:/Users/acer/Downloads/My Dataset/test/Pothole/*.jpg") + \
                     glob.glob("C:/Users/acer/Downloads/My Dataset/test/Pothole/*.jpeg") + \
                     glob.glob("C:/Users/acer/Downloads/My Dataset/test/Pothole/*.png")

plain_test_files = glob.glob("C:/Users/acer/Downloads/My Dataset/test/Plain/*.jpg") + \
                   glob.glob("C:/Users/acer/Downloads/My Dataset/test/Plain/*.jpeg") + \
                   glob.glob("C:/Users/acer/Downloads/My Dataset/test/Plain/*.png")

test_pothole = load_images(pothole_test_files)
test_plain = load_images(plain_test_files)

# ------------------------
# Prepare dataset arrays
# ------------------------
X_train = np.concatenate((train_pothole, train_plain), axis=0)
y_train = np.concatenate((
    np.ones(train_pothole.shape[0], dtype=int),
    np.zeros(train_plain.shape[0], dtype=int)
), axis=0)

X_test = np.concatenate((test_pothole, test_plain), axis=0)
y_test = np.concatenate((
    np.ones(test_pothole.shape[0], dtype=int),
    np.zeros(test_plain.shape[0], dtype=int)
), axis=0)

# Shuffle datasets
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

# Reshape for CNN input
X_train = X_train.reshape(X_train.shape[0], size, size, 1)
X_test = X_test.reshape(X_test.shape[0], size, size, 1)

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# ------------------------
# Define CNN model
# ------------------------
def kerasModel4():
    model = Sequential()
    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='valid', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (5, 5), padding="same"))
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

# ------------------------
# Compile and train model
# ------------------------
model = kerasModel4()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model (increase epochs for larger dataset)
history = model.fit(X_train, y_train, epochs=50, validation_split=0.1)

# ------------------------
# Evaluate model
# ------------------------
metrics = model.evaluate(X_test, y_test)
for i, name in enumerate(model.metrics_names):
    print(f"{name}: {metrics[i]}")

# ------------------------
# Save model
# ------------------------
model.save('sample_manual.h5')  # full model
print("Model saved as sample_manual.h5")
