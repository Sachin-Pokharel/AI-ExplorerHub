from transformers import AutoTokenizer, AutoModelForSequenceClassification, PushToHubMixin
from config import Config

class BasePush(PushToHubMixin):
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name
    
    def push(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.push_to_hub(repo_name=self.model_name, organization="your-organization", token="your-token")
