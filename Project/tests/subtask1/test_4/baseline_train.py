import numpy as np
import pandas as pd
import torch
import yaml
import wandb
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight

from tests.subtask1.test_1.dataset import PolarizationDataset


from transformers import (
    TrainingArguments,
    DataCollatorWithPadding
)

from tests.subtask1.test_1.evaluate import compute_metrics
from tests.subtask1.test_1.model import get_model_tokenizer
from tests.subtask1.test_1.trainer import WeightedTrainer

CFG_PATH = "config.yaml"


def load_cfg(path=CFG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)




def train(language_file):

    cfg = load_cfg()
    wandb.init(mode="disabled")

    train = pd.read_csv(f'../../data/dev_phase/subtask1/train/{language_file}')

    model, tokenizer = get_model_tokenizer(cfg['model_name'], cfg['num_labels'])

    train_df, val_df = train_test_split(
        train,
        test_size=0.2,
        stratify=train['polarization'],
        random_state=42
    )

    train_dataset = PolarizationDataset(train_df['text'].tolist(), train_df['polarization'].tolist(), tokenizer)
    val_dataset = PolarizationDataset(val_df['text'].tolist(), val_df['polarization'].tolist(), tokenizer)

    training_args = TrainingArguments(
            output_dir=f"../../",
            num_train_epochs=cfg.get("num_epochs", 3),
            learning_rate=cfg.get("learning_rate", 2e-5),
            per_device_train_batch_size=cfg.get("per_device_train_batch_size", 64),
            per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 8),
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="eval_f1_macro",
            load_best_model_at_end=True,
            greater_is_better=True,
            save_total_limit=1,
            logging_steps=100,
            disable_tqdm=False,
            fp16=True,
        )

    labels = train["polarization"].values



    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    print(class_weights)

    class_weights = torch.tensor(class_weights, dtype=torch.float)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
        class_weights=class_weights,

    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Macro F1 score on validation set: {eval_results['eval_f1_macro']}")




    return eval_results