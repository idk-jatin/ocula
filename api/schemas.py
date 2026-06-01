from pydantic import BaseModel,Field
from typing import Optional

class PredictRequest(BaseModel):
    text: str = Field(...,min_length=1,max_length=5000,description="Raw text to classify")
    lang: Optional[str] = Field("auto",description="Language hint: en / hi / hi-Latn / auto")

class ExplainRequest(BaseModel):
    text:str = Field(...,min_length=1,max_length=2000,description="Explanation text")
    lang: Optional[str] = "auto"
    top_k: int = Field(10,ge=1,le=30,description="Number of top tokens to return")

class TokenScore(BaseModel):
    word: str
    score: float
    index: int

class PredictResponse(BaseModel):
    label: str
    label_id: int
    confidence: float
    probabilities : dict
    top_tokens: list[TokenScore]
    latency_ms: float

class ExplainResponse(BaseModel):
    label:str
    label_id: int
    confidence: float
    probabilities: dict
    tokens: list[TokenScore]
    highlight: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    gpu_available:bool
    gpu_name: Optional[str]
    latency_ms: float