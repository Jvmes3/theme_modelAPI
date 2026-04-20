#!/usr/bin/env python3
"""Small demo script to run the theme classifier on example texts.

It prefers a local model folder `theme_model_outputs/` (faster) and
falls back to the HF repo specified in HF_REPO_ID.
"""
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

LABELS = ["mentorship", "entrepreneurship", "startup success"]


def load_model():
    local_path = os.path.join(os.path.dirname(__file__), "theme_model_outputs")
    repo = os.getenv("HF_REPO_ID", "4nkh/theme_model")
    if os.path.isdir(local_path):
        src = local_path
    else:
        src = repo
    print(f"Loading model/tokenizer from: {src}")
    tokenizer = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src)
    return tokenizer, model


def predict(texts, threshold=0.5):
    tokenizer, model = load_model()
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    for t, p, pr in zip(texts, probs, preds):
        print("INPUT:", t)
        print("PROBS:", p)
        print("PREDS:", pr)
        print("LABELS:", [LABELS[i] for i,v in enumerate(pr) if v])
        print("-")


if __name__ == "__main__":
    examples = [
        "Our co-op paired first-time founders with veteran shop owners to troubleshoot setbacks and model problem-solving.",
        "They used preorders to validate demand and managed inventory to grow a small business."
    ]
    predict(examples)
