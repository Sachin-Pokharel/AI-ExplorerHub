import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config

class BaseInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
    
    def predict(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).tolist()[0]
        return probabilities
