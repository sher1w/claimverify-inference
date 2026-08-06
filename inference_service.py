from fastapi import FastAPI
from pydantic import BaseModel
from transformers import PreTrainedTokenizerFast, AutoModelForSequenceClassification
import torch

app = FastAPI()

MODEL_PATH = "mazarellosherwin/claimverify-indicbert-konkani"

print("Loading tokenizer...")
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
print("Tokenizer loaded")

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Model loaded")


class ClaimInput(BaseModel):
    text: str


@app.post("/predict")
def predict(input: ClaimInput):
    inputs = tokenizer(input.text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()
    verdict = "FAKE" if pred == 1 else "REAL"
    return {"verdict": verdict, "confidence": round(confidence, 4)}