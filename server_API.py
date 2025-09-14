import io
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model

# ------------------------
app = FastAPI(title="Fake/Real Report Classifier")

# ------------------------
# Load trained H5 model
model_path = r"C:\Users\acer\Downloads\pothole-detection-system-using-convolution-neural-networks-master\pothole-detection-system-using-convolution-neural-networks-master\sample_manual_augmented.h5"
size = 100
model = load_model(model_path)
print("Model loaded successfully!")

# ------------------------
@app.get("/")
def root():
    return {"message": "Fake/Real Report Classifier API is running."}

# ------------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Read uploaded file
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return JSONResponse(content={"error": "Invalid image"}, status_code=400)

        # Preprocess image
        img = cv2.resize(img, (size, size))
        img = img.astype("float32") / 255.0
        img = img.reshape(1, size, size, 1)

        # Predict
        pred = model.predict(img)
        pred_class = int(np.argmax(pred, axis=1)[0])
        confidence = float(np.max(pred))

        # Map to labels
        label = "Correct report" if pred_class == 1 else "False report"

        return {"prediction": label, "confidence": confidence}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
