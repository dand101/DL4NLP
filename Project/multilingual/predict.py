import os
import shutil
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DebertaV2Tokenizer, Trainer

from utils.funcs import load_and_tag

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./models/xlm-roberta-large-se2026s1"
dev_csv_path = "/home/teo/semeval9/data/dev_phase/subtask1/dev.csv"

model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model_checkpoint = "FacebookAI/xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
trainer = Trainer(model=model)

dev_df = load_and_tag("/home/teo/semeval9/data/dev_phase/subtask1/dev", "dev")

def preprocess_inference(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

dev_dataset = Dataset.from_pandas(dev_df)
tokenized_dev = dev_dataset.map(preprocess_inference, batched=True)

cols_to_drop = [col for col in tokenized_dev.column_names if col not in ["input_ids", "attention_mask"]]
tokenized_dev_clean = tokenized_dev.remove_columns(cols_to_drop)

model.eval() 
with torch.no_grad():
    raw_preds = trainer.predict(tokenized_dev_clean)
    predictions = np.argmax(raw_preds.predictions, axis=-1)

dev_df['polarization'] = predictions

base_sub_dir = "submission_package"
subtask_dir = os.path.join(base_sub_dir, "subtask_1")

if os.path.exists(base_sub_dir):
    shutil.rmtree(base_sub_dir)
os.makedirs(subtask_dir)

for lang_code in dev_df['lang'].unique():
    lang_subset = dev_df[dev_df['lang'] == lang_code][['id', 'polarization']]
    file_path = os.path.join(subtask_dir, f"pred_{lang_code}.csv")
    lang_subset.to_csv(file_path, index=False)

shutil.make_archive("submission2_subtask1", 'zip', base_sub_dir)