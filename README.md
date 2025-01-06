# Financial Market Sentiment Analysis with FinBERT

## Overview
This machine learning project fine-tunes the `FinBERT` model on financial news data to classify sentiment into negative, neutral, and positive categories. The fine-tuned model can be used to analyze the sentiment of financial market news, providing valuable insights for applications such as sentiment analysis, financial decision-making, and trend analysis. Beuatiful soup webscraping is done to obtain the lastest yahoo finance stock market news headlines and then current market sentiment will be determined using the trained model, and these latest headlines.
The fine-tuned model achieves an 86% accuracy in classifying sentiment, demonstrating its effectiveness in analyzing financial news and trends.

## Example Output

1. Negative: Michael Saylor's MicroStrategy arbitrage seals suggest underlying market risks
2. Negative: These Sectors Led Wednesday's Fed-Fueled Stock Market Sell-Off
3. Negative: Stock market today: Dow, S&P 500, Nasdaq clobbered as Fed, Powell signal fewer rate cuts in 2025
4. Positive: Fed's Powell Shocks Markets After Interest Rate Cut: 'It's A New Phase'
5. Negative: Brazil Traders ‘Sell First, Ask Later’ as Panic Hits Markets
6. Negative: US rate futures price in Fed on hold in January, less than two cuts in 2025
7. Positive: Rocket Pharmaceuticals Stock Soars as Jefferies Initiates Coverage at 'Buy'
8. Neutral: Focus on relationships is the key to navigating market downturns
9. Positive: Analyst unveils top tech stocks to buy for 2025
10. Positive: Equity Markets Higher Ahead of Fed Decision
11. Neutral: Tech giants, Wall Street, and corporate elite are pouring millions into Donald Trump’s inauguration
12. Neutral: Why Is Corvus Pharmaceuticals Stock Trading Lower On Wednesday?
13. Positive: Freight market shows additional signs of recovery
14. Neutral: What's Going On With Viking Therapeutics Stock On Wednesday?
15. Neutral: A 1130% Share Price Rally Draws Questions From Brazil’s Stock Exchange Operator
16. Positive: Pharma’s data and analytics market forecast to reach $2.1bn by 2028
17. Positive: Top Stock Movers Now: Jabil, Nvidia, General Mills, and More
18. Negative: Here’s Why the Dow Just Had Its Worst Slump in 50 Years
19. Neutral: Nvidia stock pops 3% after slumping into correction territory
20. Negative: Oklo Stock Jumps, Retreats in Whipsaw Trading After Switch Announcement
21. Positive: Jabil Stock Soars on Circuit Board Maker's Q1 Results, Raised Outlook


Sentiment Counts:
sentiment
neutral     6
positive    8
negative    7
Name: count, dtype: int64

Overall Average Sentiment Score: 1.05 (neutral)


## Requirements
- Python 3.6+
- Transformers library (`transformers`) version `4.17.0`
- Datasets library (`datasets`) version `1.18.3`
- Scikit-learn library (`sklearn`) version `0.24.2`
- PyTorch library (`torch`) version `1.9.1`
- BeautifulSoup library (`beautifulsoup4`) version `4.9.3`
- Requests library (`requests`) version `2.25.1`

You can install the required packages using:
pip install -r requirements.txt

## Dataset
The project uses a financial news dataset stored in a CSV file. The dataset contains headlines and a sentiment column, which classifies each headline as either negative, neutral, or positive. The dataset is pre-processed into processed_data, which contains two columns: headline and label. The sentiment labels are mapped as follows:
positive → 2
neutral → 1
negative → 0


## Fine-Tuning with FinBERT

The script fine_tune_model.py fine-tunes the yiyanghkust/finbert-tone model using the specified dataset and saves the fine-tuned model in the models/financial_sentiment_model directory.


## How to Run

python .\src\main.py

The script fine_tune_model.py fine-tunes the yiyanghkust/finbert-tone model using the specified dataset and saves the fine-tuned model in the models/financial_sentiment_model directory.

## Results

After training, the script evaluates the model on the test set, calculates accuracy, and prints the classification report, including key metrics such as precision, recall, F1-score, and accuracy.


Accuracy: 0.8629

{
  "negative": {
    "precision": 0.8051,
    "recall": 0.8796,
    "f1-score": 0.8407,
    "support": 108.0
  },
  "neutral": {
    "precision": 0.8954,
    "recall": 0.8891,
    "f1-score": 0.8922,
    "support": 568.0
  },
  "positive": {
    "precision": 0.8229,
    "recall": 0.8061,
    "f1-score": 0.8144,
    "support": 294.0
  },
}


## Future Work

In future iterations, I plan to create a custom dataset that is more relevant to the markets I am focused on. This dataset will be specifically tailored to include financial news headlines from outlets I am concerned with such as CNBC, Yahoo Finance, Bloomberg, and others. Additionally, I will target data from major stock exchanges like the US NASDAQ and NYSE. This approach will allow for more accurate sentiment analysis for the markets I am interested in and improve the model's ability to track and predict market trends based on real-time financial news.

