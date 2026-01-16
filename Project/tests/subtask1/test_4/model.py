import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput


class RobertaSbertFusion(nn.Module):
    def __init__(
        self,
        roberta_name: str,
        sbert_name: str,
        num_labels: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Encoders
        self.roberta = AutoModel.from_pretrained(roberta_name)
        self.sbert = AutoModel.from_pretrained(sbert_name)

        self.config = self.roberta.config


        # Sizes
        roberta_dim = self.roberta.config.hidden_size
        sbert_dim = self.sbert.config.hidden_size

        # Fusion head
        self.classifier = nn.Sequential(
            nn.Linear(roberta_dim + sbert_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels)
        )

    @staticmethod
    def mean_pooling(last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        rob_out = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        rob_cls = rob_out.last_hidden_state[:, 0]  # CLS

        sb_out = self.sbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        sb_emb = self.mean_pooling(
            sb_out.last_hidden_state,
            attention_mask
        )

        fused = torch.cat([rob_cls, sb_emb], dim=1)
        logits = self.classifier(fused)

        return SequenceClassifierOutput(logits=logits)


def get_model_and_tokenizer(model_name: str, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained("roberta-large", use_fast=True)
    model = RobertaSbertFusion(
        roberta_name="roberta-large",
        sbert_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        num_labels=2
    )
    return model, tokenizer