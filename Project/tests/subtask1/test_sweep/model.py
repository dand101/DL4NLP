from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_model_and_tokenizer(model_name: str, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer
