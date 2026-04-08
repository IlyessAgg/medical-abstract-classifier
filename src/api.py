from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from src.model import get_embeddings

app = FastAPI(title="Medical Abstract Classifier")

# Load model and encoder at startup
classifier = joblib.load("models/classifier.joblib")
encoder = joblib.load("models/encoder.joblib")

class PredictionRequest(BaseModel):
    question: str
    context: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    text = request.question + ' ' + request.context
    emb = get_embeddings([text])
    probs = classifier.predict_proba(emb)
    confidence = np.max(probs)
    label = encoder.inverse_transform([np.argmax(probs)])
    return PredictionResponse(label=label[0], confidence=confidence)