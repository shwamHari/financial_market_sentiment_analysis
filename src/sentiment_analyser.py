import pandas as pd

# Load the dataset
df = pd.read_csv("data/labelled_headline_data.csv")

# Map sentiments to integers
sentiment_map = {"positive": 2, "neutral": 1, "negative": 0}
df['label'] = df['sentiment'].map(sentiment_map)

# Drop the original sentiment column
df = df[['label', 'headline']]

# Save processed data
df.to_csv("data/processed_labelled_headline_data.csv", index=False)
