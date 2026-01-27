from pydantic import BaseModel
from typing import Dict


class PredictionResponse(BaseModel):
    status: str
    image_type: str
    detected_disease: str
    confidence: float
    all_probabilities: Dict[str, float]
