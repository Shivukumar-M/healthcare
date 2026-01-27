from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import json

# -----------------------------
# APP SETUP
# -----------------------------
app = FastAPI(title="Healthcare AI – Multi Disease Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LOAD LABELS
# -----------------------------
with open("labels.json", "r") as f:
    LABELS = json.load(f)

# -----------------------------
# LOAD MODEL (KERAS 3 SAFE)
# -----------------------------
MODEL_PATH = "model/saved_model"

model = keras.layers.TFSMLayer(
    MODEL_PATH,
    call_endpoint="serving_default"
)

# Wrap as callable model
def predict_fn(inputs):
    outputs = model(inputs)
    # SavedModel returns dict: get first value
    return list(outputs.values())[0]

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# -----------------------------
# API ENDPOINT
# -----------------------------
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    input_tensor = preprocess_image(image)

    preds = predict_fn(input_tensor)[0]
    idx = int(np.argmax(preds))

    return {
        "status": "success",
        "detected_disease": LABELS[idx],
        "confidence": round(float(preds[idx]) * 100, 2),
        "all_probabilities": {
            LABELS[i]: round(float(preds[i]) * 100, 2)
            for i in range(len(LABELS))
        }
    }
