from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class SentenceEmbeddingClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = self.encoder.config
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def mean_pooling(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        sentence_emb = self.mean_pooling(
            outputs.last_hidden_state,
            attention_mask
        )

        logits = self.classifier(sentence_emb)

        return SequenceClassifierOutput(logits=logits)

def get_model_and_tokenizer(model_name: str, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = SentenceEmbeddingClassifier(model_name, num_labels)
    return model, tokenizer



