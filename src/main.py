import os
from scraper import *
from preprocess_data import preprocess_data
from preprocess_sentfin import preprocess_financial_data
from fine_tune_model import fine_tune_model
import subprocess


if __name__ == "__main__":
    print("Running financial sentiment analysis pipeline...")

    # Step 1: Define file paths
    labelled_headline_data = "data/labelled_headline_data.csv"
    sentfin_data = "data/sentfin.csv"
    processed_data_path = "data/processed_data.csv"

    # Step 2: Check if processed_data.csv exists and delete if present
    if os.path.exists(processed_data_path):
        os.remove(processed_data_path)
        print("Existing processed_data.csv cleared.")

    # Step 3: Preprocess both datasets and concatenate into processed_data.csv
    preprocess_financial_data(sentfin_data, processed_data_path)
    preprocess_data(labelled_headline_data, processed_data_path)

    # Step 4: Fine-tune model
    model_output_dir = "models/financial_sentiment_model"
    fine_tune_model(processed_data_path, "yiyanghkust/finbert-tone", model_output_dir)

    # Step 5: Obtain and save latest stock market headlines from yahoo finance
    url = 'https://finance.yahoo.com/topic/stock-market-news/'
    class_name = "clamp yf-18q3fnf"
    headlines = get_financial_headlines(url, class_name)

    # Save headlines to CSV
    df = pd.DataFrame({'headline': headlines})
    df.to_csv('data/yahoo_finance_headlines.csv', index=False)

    # Step 6: Analyze the scraped headlines
    subprocess.run(['python', 'extract_headlines.py'])

    print("Process completed successfully.")
