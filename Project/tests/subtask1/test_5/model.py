import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class EnglishRobertaMultiLayerCLS(nn.Module):
    """
    RoBERTa-large classifier using CLS from the last 4 layers,
    concatenated and passed through an MLP head.
    """

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.3):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        self.config = self.encoder.config
        self.num_labels = num_labels

        hidden_size = self.encoder.config.hidden_size
        cls_dim = hidden_size * 4  # last 4 layers

        self.classifier = nn.Sequential(
            nn.Linear(cls_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Take CLS token from last 4 layers
        hidden_states = outputs.hidden_states
        cls_rep = torch.cat(
            [hidden_states[i][:, 0] for i in (-1, -2, -3, -4)],
            dim=1
        )

        logits = self.classifier(cls_rep)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


def get_model_and_tokenizer(model_name: str, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = EnglishRobertaMultiLayerCLS(model_name, num_labels)
    return model, tokenizer

