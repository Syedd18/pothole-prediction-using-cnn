import numpy as np
import cv2
from keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox

# ------------------------
# Global settings
# ------------------------
size = 100

# ------------------------
# Load trained model
# ------------------------
model_path = r"C:\Users\acer\Downloads\pothole-detection-system-using-convolution-neural-networks-master\pothole-detection-system-using-convolution-neural-networks-master\sample_manual.h5"
model = load_model(model_path)

# ------------------------
# Function to predict a single image
# ------------------------
def predict_image(image_path):
    img = cv2.imread(image_path, 0)  # read in grayscale
    if img is None:
        messagebox.showerror("Error", f"Cannot read file {image_path}")
        return

    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = img.reshape(1, size, size, 1)

    pred = model.predict(img)
    pred_class = np.argmax(pred, axis=1)[0]
    label = "Pothole" if pred_class == 1 else "Plain"

    messagebox.showinfo("Prediction Result", f"Image: {image_path.split('/')[-1]}\nPredicted: {label}")

# ------------------------
# Tkinter GUI
# ------------------------
def select_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        predict_image(file_path)

root = tk.Tk()
root.title("Pothole Detector")
root.geometry("400x150")

btn = tk.Button(root, text="Select Image to Predict", command=select_image, font=("Arial", 14))
btn.pack(pady=40)

root.mainloop()
