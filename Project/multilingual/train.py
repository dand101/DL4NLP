import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import evaluate
from transformers import (
    Trainer, 
    DebertaV2Tokenizer, 
    TrainingArguments, 
    AutoModelForSequenceClassification
)
from utils.funcs import load_and_tag 
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import datetime

MODEL_CHECKPOINT = "microsoft/mdeberta-v3-base"
MAX_LENGTH = 256  
BATCH_SIZE = 16   
GRAD_ACCUM = 2
LR = 2e-5         
EPOCHS = 6        
N_FOLDS = 5       
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_df = load_and_tag("/home/teo/semeval9/data/dev_phase/subtask1/train", "train")
dev_df = load_and_tag("/home/teo/semeval9/data/dev_phase/subtask1/dev", "dev")

overlap = set(train_df['text']).intersection(set(dev_df['text']))
if len(overlap) > 0:
    train_df = train_df[~train_df['text'].isin(overlap)]

unique_langs = sorted(train_df['lang'].unique().tolist())
lang2id = {lang: i for i, lang in enumerate(unique_langs)}
id2lang = {i: lang for lang, i in lang2id.items()}


lang_stats = train_df.groupby(['lang', 'polarization']).size().unstack(fill_value=0)
weight_matrix = torch.zeros((len(unique_langs), 2))
for lang, lang_id in lang2id.items():
    n_lang = lang_stats.loc[lang].sum()
    weight_matrix[lang_id, 0] = n_lang / (2 * lang_stats.loc[lang, 0] if lang_stats.loc[lang, 0] > 0 else 1.0)
    weight_matrix[lang_id, 1] = n_lang / (2 * lang_stats.loc[lang, 1] if lang_stats.loc[lang, 1] > 0 else 1.0)
    print(f"{id2lang[lang_id]}: 1 - {lang_stats.loc[lang, 1]}, 0 - {lang_stats.loc[lang, 0]}")


quit()
weight_matrix = weight_matrix.to(device)

train_split, internal_val_split = train_test_split(
    train_df, test_size=0.1, stratify=train_df[['lang', 'polarization']], random_state=42
)

model_checkpoint = "microsoft/mdeberta-v3-base"
tokenizer = DebertaV2Tokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    result = tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")
    result["labels"] = examples["polarization"]
    result["lang_ids"] = [lang2id[l] for l in examples["lang"]]
    return result

raw_datasets = DatasetDict({
    "train": Dataset.from_pandas(train_split),
    "validation": Dataset.from_pandas(internal_val_split)
})
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

cols_to_keep = ["input_ids", "attention_mask", "labels", "lang_ids"]
for split in tokenized_datasets.keys():
    cols_to_remove = [col for col in tokenized_datasets[split].column_names if col not in cols_to_keep]
    tokenized_datasets[split] = tokenized_datasets[split].remove_columns(cols_to_remove)

class MultiLingualWeightedTrainer(Trainer):
    def __init__(self, *args, lang_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lang_weights = lang_weights
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        lang_ids = inputs.get("lang_ids")
        model_inputs = {k: v for k, v in inputs.items() if k not in ["lang_ids"]}
        
        outputs = model(**model_inputs)
        logits = outputs.get("logits")

        raw_loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        if lang_ids is not None:
            batch_weights = self.lang_weights[lang_ids, labels]
        else:
            batch_weights = torch.ones_like(raw_loss)

        loss = (raw_loss * batch_weights).mean()
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)



metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    
    f1_score = metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
    results = {"macro_f1": f1_score}

    try:
        if eval_pred.inputs is not None:
            lang_ids = eval_pred.inputs[:, -1] 
            for l_id in np.unique(lang_ids):
                lang_name = id2lang[int(l_id)]
                mask = (lang_ids == l_id)
                if mask.any():
                    lang_f1 = metric.compute(predictions=predictions[mask], references=labels[mask], average="macro")["f1"]
                    results[f"{lang_name}_f1"] = lang_f1
    except Exception as e:
        print(f"Note: Per-language metrics skipped this time: {e}")

    return results

training_arguments = TrainingArguments(
    output_dir= f"./results_mdeberta-v3-{datetime.datetime.now().timestamp()}",
    learning_rate=2e-5,
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    report_to="wandb", 
    bf16=True,
    max_grad_norm=1.0,
    remove_unused_columns=False,
    include_for_metrics=["lang_ids"]
)

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2).to(device)

trainer = MultiLingualWeightedTrainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    lang_weights=weight_matrix
)

trainer.train()