import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the fine-tuned sentiment analysis model
model_path = "models/financial_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
sentiment_analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Load scraped headlines
headlines_df = pd.read_csv('data/yahoo_finance_headlines.csv')

# Apply sentiment analysis on headlines
headlines_df['sentiment'] = headlines_df['headline'].apply(lambda x: sentiment_analyzer(x)[0]['label'].lower())

# Ensure full rows and columns are printed
pd.set_option('display.max_rows', None)  # Ensure all rows are printed
pd.set_option('display.max_colwidth', None)  # Ensure full headline text is printed

# Print sentiment and headline
print(headlines_df[['sentiment', 'headline']])

# Calculate and print sentiment counts
sentiment_counts = headlines_df['sentiment'].value_counts()
print("\nSentiment Counts:")
print(sentiment_counts)

# Calculate overall average sentiment using updated sentiment_map
sentiment_map = {"positive": 2, "neutral": 1, "negative": 0}
headlines_df['sentiment_score'] = headlines_df['sentiment'].map(sentiment_map)
average_sentiment_score = headlines_df['sentiment_score'].mean()

print(f"\nOverall Average Sentiment Score: {average_sentiment_score:.2f}")
