import pandas as pd

def preprocess_data(input_csv, output_csv):
    # Load the existing processed dataset
    try:
        df_existing = pd.read_csv(output_csv)
    except FileNotFoundError:
        df_existing = pd.DataFrame(columns=['label', 'headline'])

    # Load new data
    new_data = pd.read_csv(input_csv)

    # Normalize column names to handle case or whitespace issues
    new_data.columns = new_data.columns.str.strip().str.lower()

    # Map sentiments to numerical labels
    sentiment_map = {"positive": 2, "neutral": 1, "negative": 0}
    new_data['label'] = new_data['sentiment'].map(sentiment_map)

    # Ensure required columns exist
    if 'headline' not in new_data.columns or 'label' not in new_data.columns:
        raise KeyError("Required columns 'headline' or 'sentiment' not found in the dataset")

    # Concatenate new data to existing processed data
    df_combined = pd.concat([df_existing, new_data[['label', 'headline']]], ignore_index=True)

    # Save the combined dataset
    df_combined.to_csv(output_csv, index=False)
    print(f"Processed data appended and saved to {output_csv}")

if __name__ == "__main__":
    input_path = "data/labelled_headline_data.csv"
    output_path = "data/processed_data.csv"
    preprocess_data(input_path, output_path)
