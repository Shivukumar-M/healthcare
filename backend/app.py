from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from gemini.modality_detector import detect_modality
from services.predictor import init_models, predict


app = FastAPI(title="Healthcare AI – Hybrid Model")

# ----------------------------------------
# CORS (allow frontend access)
# ----------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------
# Load models at startup
# ----------------------------------------
@app.on_event("startup")
def startup():
    init_models()


# ----------------------------------------
# Analyze endpoint
# ----------------------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Validate image
    try:
        Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {
            "status": "error",
            "message": "Invalid image file"
        }

    # Step 1: Detect modality via Gemini
    modality = detect_modality(image_bytes)

    if modality == "unknown":
        return {
            "status": "error",
            "message": "Unable to detect medical image type"
        }

    # Step 2: CNN prediction (bytes → preprocess inside predictor)
    result = predict(modality, image_bytes)

    return {
        "status": "success",
        "detected_image_type": modality,
        "detected_disease": result["detected_disease"],
        "confidence": result["confidence"],
        "all_probabilities": result["all_probabilities"],
        "medical_note": (
            "This AI analysis is for assistance only and "
            "is not a medical diagnosis. Please consult a doctor."
        )
    }
