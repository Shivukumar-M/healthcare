import time
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL

# ----------------------------------------
# Gemini setup
# ----------------------------------------
genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel(GEMINI_MODEL)

# ----------------------------------------
# In-memory cache
# key = "imageType_disease"
# value = explanation text
# ----------------------------------------
EXPLANATION_CACHE = {}

# Cache expiry (seconds)
CACHE_TTL = 60 * 60  # 1 hour


def generate_explanation(image_type: str, disease: str, confidence: float) -> str:
    """
    Generates a doctor-like, patient-friendly explanation.
    Uses caching to reduce Gemini API calls.
    """

    cache_key = f"{image_type}_{disease}"

    # ----------------------------------------
    # 1. Return cached explanation if valid
    # ----------------------------------------
    if cache_key in EXPLANATION_CACHE:
        cached = EXPLANATION_CACHE[cache_key]
        if time.time() - cached["time"] < CACHE_TTL:
            return cached["text"]
        del EXPLANATION_CACHE[cache_key]

    # ----------------------------------------
    # 2. Doctor-style prompt
    # ----------------------------------------
    prompt = f"""
You are an experienced medical doctor speaking directly to a patient.

Explain the result in very simple, reassuring language.

Details:
- Image type: {image_type}
- Condition detected: {disease}
- Confidence: {confidence}%

Instructions:
- Speak like a real doctor in a clinic
- Use calm, friendly words
- No medical jargon
- No fear-inducing language
- Explain:
  1) What this condition means
  2) How serious it usually is
  3) What the patient should do next
- Use 4–5 short bullet points
- End with a gentle medical disclaimer
"""

    # ----------------------------------------
    # 3. Call Gemini safely
    # ----------------------------------------
    try:
        response = MODEL.generate_content(prompt)

        if not response or not response.text:
            raise RuntimeError("Empty Gemini response")

        explanation = response.text.strip()

        # ----------------------------------------
        # 4. Cache result
        # ----------------------------------------
        EXPLANATION_CACHE[cache_key] = {
            "text": explanation,
            "time": time.time()
        }

        return explanation

    except Exception as e:
        print("Gemini explanation error:", e)

        # ----------------------------------------
        # 5. Doctor-style fallback (NO GEMINI)
        # ----------------------------------------
        seriousness = "not usually dangerous" if confidence < 70 else "something we should look at carefully"

        return (
            f"• Based on the image, we see signs related to **{disease}**.\n"
            f"• In most cases, this condition is {seriousness}.\n"
            "• The confidence level suggests the AI is reasonably sure about this result.\n"
            "• I recommend discussing this report with a qualified doctor for proper evaluation.\n"
            "• This result is meant to assist, not replace, a medical diagnosis."
        )
