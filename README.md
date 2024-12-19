# Financial Market Sentiment Analysis with FinBERT

## Overview
This project fine-tunes the `FinBERT` model on financial news data to classify sentiment into negative, neutral, and positive categories. The fine-tuned model can be used to analyze the sentiment of financial market news, providing valuable insights for applications such as sentiment analysis, financial decision-making, and trend analysis. Beuatiful soup webscraping is done to obtain the lastest yahoo finance stock market news headlines and then current market sentiment will be determined using the trained model, and thse latest headlines.

## Requirements
- Python 3.6+
- Transformers library (`transformers`)
- Datasets library (`datasets`)
- Scikit-learn library (`sklearn`)

You can install the required packages using:
pip install transformers datasets scikit-learn

## Dataset
The project uses a financial news dataset stored in a CSV file. The dataset contains headlines and a sentiment column, which classifies each headline as either negative, neutral, or positive. The dataset is pre-processed into processed_data, which contains two columns: headline and label. The sentiment labels are mapped as follows:
positive → 2
neutral → 1
negative → 0


## Fine-Tuning with FinBERT

The script fine_tune_model.py fine-tunes the yiyanghkust/finbert-tone model using the specified dataset and saves the fine-tuned model in the models/financial_sentiment_model directory.


## How to Run

python main.py

The script fine_tune_model.py fine-tunes the yiyanghkust/finbert-tone model using the specified dataset and saves the fine-tuned model in the models/financial_sentiment_model directory.

## Results

After training, the script evaluates the model on the test set, calculates accuracy, and prints the classification report, including key metrics such as precision, recall, F1-score, and accuracy.


## Evaluation Metrics:
Accuracy: The overall accuracy of the model.
F1-Score: The weighted F1-score for evaluating the model’s performance across all sentiment classes (negative, neutral, positive).



original
Accuracy: 0.8629
{'negative': {'precision': 0.8050847457627118, 'recall': 0.8796296296296297, 'f1-score': 0.8407079646017699, 'support': 108.0}, 'neutral': {'precision': 0.8953900709219859, 'recall': 0.8890845070422535, 'f1-score': 0.892226148409894, 'support': 568.0}, 'positive': {'precision': 0.8229166666666666, 'recall': 0.8061224489795918, 'f1-score': 0.8144329896907216, 'support': 294.0}, 'accuracy': 0.8628865979381444, 'macro avg': {'precision': 0.8411304944504548, 'recall': 0.8582788618838251, 'f1-score': 0.8491223675674618, 'support': 970.0}, 'weighted avg': {'precision': 0.8633692915732586, 'recall': 0.8628865979381444, 'f1-score': 0.8629115581885394, 'support': 970.0}}



sentfin
Accuracy: 0.8678
{'negative': {'precision': 0.8600405679513184, 'recall': 0.8907563025210085, 'f1-score': 0.8751289989680082, 'support': 476.0}, 'neutral': {'precision': 0.8611111111111112, 'recall': 0.8318425760286225, 'f1-score': 0.8462238398544131, 'support': 559.0}, 'positive': {'precision': 0.8813868613138686, 'recall': 0.8846153846153846, 'f1-score': 0.8829981718464351, 'support': 546.0}, 'accuracy': 0.8678051865907653, 'macro avg': {'precision': 0.8675128467920995, 'recall': 0.8690714210550051, 'f1-score': 0.8681170035562854, 'support': 1581.0}, 'weighted avg': {'precision': 0.8677910485346685, 'recall': 0.8678051865907653, 'f1-score': 0.8676265223374716, 'support': 1581.0}}