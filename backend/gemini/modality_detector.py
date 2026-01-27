import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL

genai.configure(api_key=GEMINI_API_KEY)

MODEL = genai.GenerativeModel(GEMINI_MODEL)


def detect_modality(image_bytes: bytes) -> str:
    """
    Detects the medical image modality using Gemini.

    Returns one of:
    - "brain"
    - "breast"
    - "chest"
    - "skin"
    - "unknown"
    """

    prompt = """
You are a medical imaging classifier.

Given ONE medical image, classify it into EXACTLY ONE of the following categories:

- brain_mri
- breast_histopathology
- chest_xray
- skin_image

Rules:
- Return ONLY the category name.
- Do NOT explain.
- Do NOT add extra text.
- If unsure, return "unknown".
"""

    try:
        response = MODEL.generate_content(
            [
                prompt,
                {
                    "mime_type": "image/jpeg",
                    "data": image_bytes
                }
            ]
        )

        if not response.text:
            return "unknown"

        result = response.text.strip().lower()

        if result == "brain_mri":
            return "brain"
        if result == "breast_histopathology":
            return "breast"
        if result == "chest_xray":
            return "chest"
        if result == "skin_image":
            return "skin"

        return "unknown"

    except Exception as e:
        print("Gemini modality detection error:", str(e))
        return "unknown"
