import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import evaluate
import gc
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset, DatasetDict
from transformers import (
    Trainer, 
    DebertaV2Tokenizer, 
    TrainingArguments, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

MODEL_CHECKPOINT = "microsoft/mdeberta-v3-base"
MAX_LENGTH = 128
BATCH_SIZE = 8       
GRAD_ACCUMULATION = 4
LEARNING_RATE = 1e-5 
EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
NUM_FOLDS = 5
OUTPUT_DIR = f"./results_stratified_aug_"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on {device} with model {MODEL_CHECKPOINT}")

def load_and_tag(folder_path, split_name):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    dfs = []
    for f in files:
        filename = os.path.basename(f).split('.')[0]
        lang = filename        
        if "_augumented" in filename:
            lang = filename.replace("_augumented", "")
        
        df = pd.read_csv(f)
        df['lang'] = lang
        df['split'] = split_name
        dfs.append(df)
    
    if not dfs:
        raise ValueError(f"No CSV files found in {folder_path}")
        
    return pd.concat(dfs, ignore_index=True)

class FocalLossTrainer(Trainer):
    def __init__(self, *args, alpha=1, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "lang_ids"})
        logits = outputs.get("logits")
        
        ce_loss = self.loss_fct(logits, labels)
        
        pt = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        loss = focal_loss.mean()
        
        return (loss, outputs) if return_outputs else loss

metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="macro")

tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_CHECKPOINT)

def preprocess_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=MAX_LENGTH, 
        padding=False 
    )
    tokenized_inputs["labels"] = examples["polarization"]
    return tokenized_inputs


def run_training():
    train_path = "/home/teo/semeval9/data/dev_phase/subtask1/train_augmented"
    full_df = load_and_tag(train_path, "train")
    

    
    original_path = "/home/teo/semeval9/data/dev_phase/subtask1/train" 
    original_df = load_and_tag(original_path, "train")
    real_ids = set(original_df['id'].unique())
    
    full_df['is_real'] = full_df['id'].isin(real_ids)
    
    print(f"Total rows: {len(full_df)}")
    print(f"Real rows: {full_df['is_real'].sum()}")
    print(f"Synthetic rows: {(~full_df['is_real']).sum()}")

    full_df['stratify_col'] = full_df['lang'] + "_" + full_df['polarization'].astype(str)
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df['stratify_col'])):
        print(f"\n{'='*20} FOLD {fold+1}/{NUM_FOLDS} {'='*20}")
        
        train_df = full_df.iloc[train_idx]
        val_df = full_df.iloc[val_idx]
        

        val_df = val_df[val_df['is_real'] == True]
        
        print(f"   Train Size: {len(train_df)} (Mixed)")
        print(f"   Val Size:   {len(val_df)} (Real Only)")

        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df)
        
        tokenized_train = train_ds.map(preprocess_function, batched=True)
        tokenized_val = val_ds.map(preprocess_function, batched=True)
        
        cols = ["input_ids", "attention_mask", "labels"]
        tokenized_train = tokenized_train.remove_columns([c for c in tokenized_train.column_names if c not in cols])
        tokenized_val = tokenized_val.remove_columns([c for c in tokenized_val.column_names if c not in cols])

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=2)
        model = model.to(device)

        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/fold_{fold+1}",
            learning_rate=2e-5,        
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            gradient_accumulation_steps=GRAD_ACCUMULATION,
            num_train_epochs=4,          
            weight_decay=0.01,       
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            bf16=True, 
            report_to="wandb",
            group_by_length=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()
        trainer.save_model(f"{OUTPUT_DIR}/fold_{fold+1}/best_model")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}/fold_{fold+1}/best_model")
        
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    run_training()