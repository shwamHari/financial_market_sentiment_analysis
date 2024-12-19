import pandas as pd
import json

def preprocess_financial_data(input_csv="data/sentfin.csv", output_csv="data/processed_data.csv"):
    # Load the dataset
    df = pd.read_csv(input_csv)

    # Function to extract label from the Decisions column
    def extract_label(decisions):
        try:
            # Parse the JSON-like string
            decisions_dict = json.loads(decisions)
            # Include only rows with a single sentiment rating
            if len(decisions_dict) == 1:
                sentiment = list(decisions_dict.values())[0].lower()
                # Map sentiment to integers
                return {"positive": 2, "neutral": 1, "negative": 0}.get(sentiment)
        except (json.JSONDecodeError, TypeError):
            # Handle invalid JSON or other issues gracefully
            return None

    # Apply the function to extract labels
    df['label'] = df['Decisions'].apply(extract_label)

    # Keep only rows with valid labels (non-null values)
    processed_df = df[['label', 'Title']].dropna()

    # Convert the 'label' column to integers explicitly
    processed_df['label'] = processed_df['label'].astype(int)

    # Rename columns to match the desired output format
    processed_df.rename(columns={"Title": "headline"}, inplace=True)

    # Save the processed data to a new CSV
    processed_df.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    preprocess_financial_data()
