import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tkinter import Tk, filedialog
import os

# ------------------------
size = 100
model_path = r"C:\Users\acer\Downloads\pothole-detection-system-using-convolution-neural-networks-master\pothole-detection-system-using-convolution-neural-networks-master\sample_manual_augmented.h5"

# ------------------------
# Load model
model = load_model(model_path)
print("Model loaded successfully!")

# ------------------------
# Tkinter file selection
root = Tk()
root.withdraw()  # Hide main window

file_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("All image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
)

if not file_path:
    print("No file selected.")
    exit()

# ------------------------
# Load and preprocess image
img = cv2.imread(file_path, 0)  # Grayscale
if img is None:
    print("Cannot read the selected image.")
    exit()

img = cv2.resize(img, (size, size))
img = img.astype('float32') / 255.0
img = img.reshape(1, size, size, 1)

# ------------------------
# Predict
pred = model.predict(img)
pred_class = np.argmax(pred, axis=1)[0]

print(f"Prediction: {'Pothole' if pred_class == 1 else 'Plain road'}")
