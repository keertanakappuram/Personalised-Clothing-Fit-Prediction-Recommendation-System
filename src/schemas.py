"""
schemas.py — Pydantic Request/Response Schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class PredictRequest(BaseModel):
    height: Optional[str] = Field(None, example="5'6\"")
    weight_lbs: Optional[float] = Field(None, example=140.0)
    age: Optional[float] = Field(None, example=32.0)
    size: Optional[str] = Field(None, example="M")
    category: Optional[str] = Field(None, example="fitted_dress")
    body_type: Optional[str] = Field(None, example="hourglass")
    review_text: Optional[str] = Field(None, example="Usually wear a medium but this ran tight")
    rating: Optional[float] = Field(None, example=8.0)
    bust_size: Optional[str] = Field(None, example="34B")

    class Config:
        json_schema_extra = {
            "example": {
                "height": "5'6\"",
                "weight_lbs": 140,
                "age": 32,
                "size": "M",
                "category": "fitted_dress",
                "body_type": "hourglass",
                "review_text": "Usually wear a medium but this ran a bit tight",
                "rating": 8,
                "bust_size": "34B"
            }
        }

class PredictResponse(BaseModel):
    fit: str = Field(..., example="small")
    confidence: float = Field(..., example=0.81)
    probabilities: Dict[str, float] = Field(
        ..., example={"small": 0.81, "fit": 0.15, "large": 0.04}
    )
    latency_ms: Optional[float] = None

class BatchRequest(BaseModel):
    items: List[PredictRequest]

class BatchResponse(BaseModel):
    predictions: List[PredictResponse]
    count: int
    latency_ms: Optional[float] = None
