from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    XLMRobertaForSequenceClassification
)


def get_model_tokenizer(model_name, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer



