import base64
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL

genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel(GEMINI_MODEL)


def detect_modality(image_bytes: bytes) -> str:
    prompt = """
You are a medical imaging classifier.

Classify the image into EXACTLY one:
- brain_mri
- breast_histopathology
- chest_xray
- skin_image

Rules:
- Return ONLY the category name
- No explanation
- If unsure return "unknown"
"""

    try:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        response = MODEL.generate_content([
            prompt,
            {
                "mime_type": "image/jpeg",
                "data": image_b64
            }
        ])

        if not response or not response.text:
            print("Gemini returned empty response")
            return "unknown"

        result = response.text.strip().lower()

        mapping = {
            "brain_mri": "brain",
            "breast_histopathology": "breast",
            "chest_xray": "chest",
            "skin_image": "skin"
        }

        return mapping.get(result, "unknown")

    except Exception as e:
        print("Gemini modality detection error:", e)
        return "unknown"
