from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def predict_sentiment(model_dir, headlines):
    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Tokenize input headlines
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

    # Map predictions back to labels
    label_map = {2: "positive", 1: "neutral", 0: "negative"}
    return [label_map[pred.item()] for pred in predictions]

if __name__ == "__main__":
    model_path = "models/financial_sentiment_model"
    headlines = [
        "The company's growth strategy is highly promising.",
        "The firm is facing significant challenges in production.",
        "No major changes are expected in the upcoming quarter."
    ]
    predictions = predict_sentiment(model_path, headlines)
    print("Predictions:", predictions)
