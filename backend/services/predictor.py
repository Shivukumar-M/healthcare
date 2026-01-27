import os
import json
import numpy as np
import tensorflow as tf
from typing import Dict

from config import MODEL_PATHS
from services.image_utils import preprocess_image


MODELS: Dict[str, dict] = {}


def init_models():
    """
    Load all CNN models into memory at startup.
    """
    global MODELS

    for modality, base_path in MODEL_PATHS.items():
        model_path = os.path.join(base_path, "saved_model")
        labels_path = os.path.join(base_path, "labels.json")

        if not os.path.exists(model_path):
            print(f"Model not found for {modality}, skipping")
            continue

        print(f"Loading {modality} model...")

        model = tf.keras.layers.TFSMLayer(
            model_path,
            call_endpoint="serving_default"
        )

        with open(labels_path, "r") as f:
            labels = json.load(f)

        MODELS[modality] = {
            "model": model,
            "labels": labels
        }

    print("All available models loaded")

def predict(modality: str, image_bytes: bytes) -> dict:
    """
    Run CNN prediction for a given modality.
    """
    if modality not in MODELS:
        return {
            "detected_disease": "unknown",
            "confidence": 0.0,
            "all_probabilities": {}
        }

    model = MODELS[modality]["model"]
    labels = MODELS[modality]["labels"]

    img = preprocess_image(image_bytes)

    output = model(img)

    if isinstance(output, dict):
        preds = list(output.values())[0].numpy()[0]
    else:
        preds = output.numpy()[0]

    probs = {
        labels[i]: round(float(preds[i]) * 100, 2)
        for i in range(len(labels))
    }

    detected = max(probs, key=probs.get)

    return {
        "detected_disease": detected,
        "confidence": probs[detected],
        "all_probabilities": probs
    }
