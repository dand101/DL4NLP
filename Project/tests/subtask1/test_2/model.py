from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_model_and_tokenizer(model_name: str, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer

#
# import torch
# import torch.nn as nn
# from transformers import AutoModel, AutoTokenizer
# from transformers.modeling_outputs import SequenceClassifierOutput
#
#
# class RobertaAttnPoolMSD(nn.Module):
#     def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1, msd_k: int = 5):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(model_name)
#         self.config = self.encoder.config
#         hidden = self.config.hidden_size
#
#         self.attn = nn.Linear(hidden, 1, bias=False)
#
#         self.dropout = nn.Dropout(dropout)
#         self.classifier = nn.Linear(hidden, num_labels)
#
#         self.msd_k = msd_k
#
#     def _pool(self, last_hidden_state, attention_mask):
#         scores = self.attn(last_hidden_state).squeeze(-1)
#         scores = scores.masked_fill(attention_mask == 0, -1e9)
#         w = torch.softmax(scores, dim=1)
#         pooled = torch.sum(last_hidden_state * w.unsqueeze(-1), dim=1)
#         return pooled
#
#     def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
#         out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
#         pooled = self._pool(out.last_hidden_state, attention_mask)
#
#         logits_sum = None
#
#         for _ in range(self.msd_k):
#             dropped = nn.functional.dropout(pooled, p=self.dropout.p, training=True)
#             logits = self.classifier(dropped)
#             logits_sum = logits if logits_sum is None else (logits_sum + logits)
#
#         logits_avg = logits_sum / float(self.msd_k)
#         return SequenceClassifierOutput(logits=logits_avg)
#
#
# def get_model_and_tokenizer(model_name: str, num_labels: int):
#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#     model = RobertaAttnPoolMSD(
#         model_name=model_name,
#         num_labels=num_labels,
#         dropout=0.1,
#         msd_k=5,   # try 5; 10 is slower
#     )
#     return model, tokenizer
