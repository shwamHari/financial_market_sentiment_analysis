from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import classification_report, accuracy_score, f1_score

def compute_metrics(predictions):
    preds = predictions.predictions.argmax(-1)
    true_labels = predictions.label_ids
    f1 = f1_score(true_labels, preds, average='weighted')
    return {
        'accuracy': accuracy_score(true_labels, preds),
        'f1': f1,
    }

def fine_tune_model(processed_csv, model_name, output_dir):
    # Load the dataset
    dataset = Dataset.from_csv(processed_csv)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Tokenize the data
    def tokenize_function(examples):
        return tokenizer(examples["headline"], padding="max_length", truncation=True)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Split into train and test datasets
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",  # Optional logging directory
        logging_strategy="epoch",  # Log after each epoch
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # Custom metric function
    )

    # Fine-tune the model
    trainer.train()

    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # Evaluate the model on the test set
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)  # Get the predicted labels
    true_labels = predictions.label_ids

    # Compute classification report using sklearn
    class_report = classification_report(true_labels, preds, target_names=["negative", "neutral", "positive"], output_dict=True)

    # Print accuracy and classification report
    accuracy = accuracy_score(true_labels, preds)
    print(f"Accuracy: {accuracy:.4f}")
    print(class_report)

if __name__ == "__main__":
    fine_tune_model("data/processed_data.csv", "yiyanghkust/finbert-tone", "models/financial_sentiment_model")
