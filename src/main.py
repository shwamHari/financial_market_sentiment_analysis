import os
from scraper import *
from preprocess_data import preprocess_data
from fine_tune_model import fine_tune_model
import subprocess


if __name__ == "__main__":
    print("Running financial sentiment analysis pipeline...")

    # Step 1: Preprocess data
    raw_data_path = "data/sentfin.csv"
    processed_data_path = "data/processed_data.csv"
    # preprocess_data(raw_data_path, processed_data_path)

    # Step 2: Fine-tune model
    model_output_dir = "models/financial_sentiment_model"
    fine_tune_model(processed_data_path, "yiyanghkust/finbert-tone", model_output_dir)

    url = 'https://finance.yahoo.com/'
    class_name = 'Mb(5px) D(ib) Fz(18px) Fw(600) C($c-fuji-grey-3) Mt(15px)'  # Yahoo Finance class for headlines
    headlines = get_financial_headlines(url, class_name)

    # Save headlines to CSV
    df = pd.DataFrame({'headline': headlines})
    df.to_csv('data/yahoo_finance_headlines.csv', index=False)

    # Step 4: Analyze the scraped headlines
    subprocess.run(['python', 'extract_headlines.py'])

    print("Process completed successfully.")
