import pandas as pd

def preprocess_data(input_csv, output_csv):
    # Load the dataset
    df = pd.read_csv(input_csv)

    # Normalize column names to handle case or whitespace issues
    df.columns = df.columns.str.strip().str.lower()

    # Map sentiments to numerical labels
    sentiment_map = {"positive": 2, "neutral": 1, "negative": 0}
    df['label'] = df['sentiment'].map(sentiment_map)

    # Ensure required columns exist
    if 'headline' not in df.columns or 'label' not in df.columns:
        raise KeyError("Required columns 'headline' or 'sentiment' not found in the dataset")

    # Keep only relevant columns
    df = df[['label', 'headline']]

    # Save the processed dataset
    df.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")

if __name__ == "__main__":
    input_path = "data/labelled_headline_data.csv"
    output_path = "data/processed_data.csv"
    preprocess_data(input_path, output_path)
