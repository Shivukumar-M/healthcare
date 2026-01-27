import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# 🔑 Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 🤖 Gemini model (default fallback included)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# 📂 CNN model paths
MODEL_PATHS = {
    "brain": "models/brain",
    "breast": "models/breast",
    "chest": "models/chest",
    "skin": "models/skin"
}
