from config import Config
from base.base_dataset import BaseDataset
from base.base_model import BaseModel
from base.base_inference import BaseInference
from base.base_push import BasePush

def main():
    # Load dataset
    dataset_loader = BaseDataset(Config.DATASET_PATH)
    df = dataset_loader.load_dataset()
    train_dataset = dataset_loader.preprocess_dataset(df)

    # Select and load model
    model = BaseModel(Config.MODEL_NAME)

    # Train model
    model.fine_tune(train_dataset, train_dataset)

    # Inference
    inference = BaseInference(Config.MODEL_SAVE_PATH)
    text = "your input text"
    probabilities = inference.predict(text)
    print(f"Probabilities: {probabilities}")

    # Push model to Hugging Face Hub
    push_model = BasePush(Config.MODEL_SAVE_PATH, Config.MODEL_NAME)
    push_model.push()

if __name__ == "__main__":
    main()
