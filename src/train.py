import os
from preprocess_data import preprocess_data
from fine_tune_model import fine_tune_model

if __name__ == "__main__":
    raw_data_path = "data/labelled_headline_data.csv"
    processed_data_path = "data/processed_data.csv"
    model_output_dir = "models/financial_sentiment_model"

    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Step 1: Preprocess data
    preprocess_data(raw_data_path, processed_data_path)

    # Step 2: Fine-tune model
    fine_tune_model(processed_data_path, "yiyanghkust/finbert-tone", model_output_dir)
