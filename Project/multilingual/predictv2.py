import os
import shutil
import glob
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gc
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification, 
    DebertaV2Tokenizer, 
    Trainer, 
    TrainingArguments
)

TRAIN_OUTPUT_DIR = "./results_stratified_aug_" 
DEV_DATA_PATH = "/home/teo/semeval9/data/dev_phase/subtask1/dev"
NUM_FOLDS = 5
MAX_LENGTH = 128
BATCH_SIZE = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_dev_data(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    dfs = []
    
    for f in files:
        filename = os.path.basename(f).split('.')[0]
        lang = filename.replace("_augumented", "")
        
        df = pd.read_csv(f)
        df['lang'] = lang
        if 'id' not in df.columns:
            df['id'] = df.index
            
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

tokenizer_path = os.path.join(TRAIN_OUTPUT_DIR, "fold_1", "best_model")
if not os.path.exists(tokenizer_path):
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/mdeberta-v3-base")
else:
    tokenizer = DebertaV2Tokenizer.from_pretrained(tokenizer_path)

def preprocess_inference(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=MAX_LENGTH, 
        padding="max_length"
    )

def run_inference():
    dev_df = load_dev_data(DEV_DATA_PATH)
    dev_dataset = Dataset.from_pandas(dev_df)
    
    print("Tokenizing dev set...")
    tokenized_dev = dev_dataset.map(preprocess_inference, batched=True)
    
    cols_to_keep = ["input_ids", "attention_mask"]
    tokenized_dev = tokenized_dev.remove_columns([c for c in tokenized_dev.column_names if c not in cols_to_keep])

    ensemble_probs = np.zeros((len(dev_df), 2))
    
    for fold in range(1, NUM_FOLDS + 1):
        model_path = os.path.join(TRAIN_OUTPUT_DIR, f"fold_{fold}", "best_model")
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

        trainer = Trainer(
            model=model, 
            args=TrainingArguments(
                output_dir="./temp_infer", 
                per_device_eval_batch_size=BATCH_SIZE,
                report_to="wandb"
            )
        )
        
        preds_output = trainer.predict(tokenized_dev)
        logits = torch.tensor(preds_output.predictions)
        probs = F.softmax(logits, dim=-1).numpy()
        ensemble_probs += probs
        del model
        del trainer
        del logits
        torch.cuda.empty_cache()
        gc.collect()
        
    
    avg_probs = ensemble_probs / NUM_FOLDS
    final_preds = np.argmax(avg_probs, axis=-1)
    
    dev_df['polarization'] = final_preds
    
    submission_base = "submission_ensemble_v2_aug"
    subtask_dir = os.path.join(submission_base, "subtask_1")
    
    if os.path.exists(submission_base):
        shutil.rmtree(submission_base)
    os.makedirs(subtask_dir)
    
    for lang_code in dev_df['lang'].unique():
        lang_subset = dev_df[dev_df['lang'] == lang_code].copy()
        output_df = lang_subset[['id', 'polarization']]
        file_path = os.path.join(subtask_dir, f"pred_{lang_code}.csv")
        output_df.to_csv(file_path, index=False)
        
    shutil.make_archive("submission_ensemble_final_aug_v2", 'zip', submission_base)

if __name__ == "__main__":
    run_inference()