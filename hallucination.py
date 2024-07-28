from transformers import AutoModelForSequenceClassification
import torch

def calculate_hallucination(input, text):
    model = AutoModelForSequenceClassification.from_pretrained('vectara/hallucination_evaluation_model', trust_remote_code=True)
    return model.predict([(input, text)]).item()
