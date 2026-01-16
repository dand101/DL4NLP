# import numpy as np
# import torch
# from transformers import Trainer
#
#
# class WeightedTrainer(Trainer):
#     def __init__(self, *args, class_weights=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.class_weights = class_weights
#
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.get("labels")
#         outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
#         logits = outputs.logits
#
#         if labels is None:
#             loss = outputs.loss
#         else:
#             if self.class_weights is not None:
#                 w = torch.tensor(self.class_weights, device=logits.device, dtype=torch.float)
#                 loss_fct = torch.nn.CrossEntropyLoss(weight=w)
#             else:
#                 loss_fct = torch.nn.CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
#
#         return (loss, outputs) if return_outputs else loss
#
#
# def compute_class_weights(y, num_labels=2):
#     y = np.asarray(y)
#     counts = np.bincount(y, minlength=num_labels).astype(float)
#     counts[counts == 0] = 1.0
#     inv = 1.0 / counts
#     weights = inv / inv.sum() * num_labels
#     return weights.tolist()


import numpy as np
import torch
from transformers import Trainer
from torch.optim import AdamW


def get_llrd_param_groups(
    model,
    base_lr: float,
    weight_decay: float,
    layer_decay: float = 0.95,
    head_lr_mult: float = 2.5,
):
    """
    Layer-wise Learning Rate Decay for RoBERTa-like models.
    - classifier head gets base_lr * head_lr_mult
    - top layers get ~base_lr, bottom layers get base_lr * (layer_decay ** depth)
    """
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]


    if hasattr(model, "roberta"):
        encoder = model.roberta
    elif hasattr(model, "base_model"):
        encoder = model.base_model
    elif hasattr(model, "encoder"):
        encoder = model.encoder
    else:
        raise ValueError("Could not find encoder module on model for LLRD.")

    layers = encoder.encoder.layer
    n_layers = len(layers)

    param_groups = []

    def wd_for(name: str) -> float:
        return 0.0 if any(nd in name for nd in no_decay) else float(weight_decay)

    head_lr = float(base_lr) * float(head_lr_mult)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "classifier" in name:
            param_groups.append(
                {"params": [p], "lr": head_lr, "weight_decay": wd_for(name)}
            )

    for layer_idx in range(n_layers):
        lr = float(base_lr) * (float(layer_decay) ** (n_layers - layer_idx - 1))
        layer = layers[layer_idx]
        for name, p in layer.named_parameters():
            if not p.requires_grad:
                continue
            full_name = f"encoder.layer.{layer_idx}.{name}"
            param_groups.append(
                {"params": [p], "lr": lr, "weight_decay": wd_for(full_name)}
            )

    covered = set()
    for g in param_groups:
        for p in g["params"]:
            covered.add(id(p))

    bottom_lr = float(base_lr) * (float(layer_decay) ** (n_layers))
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in covered:
            continue
        param_groups.append(
            {"params": [p], "lr": bottom_lr, "weight_decay": wd_for(name)}
        )

    return param_groups


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, llrd: bool = True,
                 layer_decay: float = 0.95, head_lr_mult: float = 2.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.llrd = llrd
        self.layer_decay = layer_decay
        self.head_lr_mult = head_lr_mult

    def create_optimizer(self):

        if self.optimizer is not None:
            return self.optimizer

        if not self.llrd:
            return super().create_optimizer()

        groups = get_llrd_param_groups(
            model=self.model,
            base_lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            layer_decay=self.layer_decay,
            head_lr_mult=self.head_lr_mult,
        )

        self.optimizer = AdamW(
            groups,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        print("Optimizer param groups:", len(self.optimizer.param_groups))
        print("Top group lr sample:", self.optimizer.param_groups[0]["lr"])

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        if labels is None:
            loss = outputs.loss
        else:
            if self.class_weights is not None:
                w = torch.tensor(self.class_weights, device=logits.device, dtype=torch.float)
                loss_fct = torch.nn.CrossEntropyLoss(weight=w)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def compute_class_weights(y, num_labels=2):
    y = np.asarray(y)
    counts = np.bincount(y, minlength=num_labels).astype(float)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv / inv.sum() * num_labels
    return weights.tolist()
