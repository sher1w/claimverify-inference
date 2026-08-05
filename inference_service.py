from fastapi import FastAPI
from pydantic import BaseModel
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import PreTrainedTokenizerFast
import numpy as np

app = FastAPI()

MODEL_PATH = "mazarellosherwin/claimverify-indicbert-konkani-onnx"

print("Loading tokenizer...")
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
print("Tokenizer loaded")

print("Loading model...")
model = ORTModelForSequenceClassification.from_pretrained(MODEL_PATH, file_name="model_quantized.onnx")
print("Model loaded")


class ClaimInput(BaseModel):
    text: str


@app.post("/predict")
def predict(input: ClaimInput):
    inputs = tokenizer(input.text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()
    pred = int(np.argmax(logits, axis=1)[0])
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    confidence = float(probs[0][pred])
    verdict = "FAKE" if pred == 1 else "REAL"
    return {"verdict": verdict, "confidence": round(confidence, 4)}