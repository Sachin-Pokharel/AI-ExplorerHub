import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from config import Config

class BaseModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer, self.model = self.load_model()
    
    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        return tokenizer, model
    
    def fine_tune(self, train_dataset, eval_dataset):
        raise NotImplementedError
    
    def compute_metrics(self, eval_pred):
        raise NotImplementedError
