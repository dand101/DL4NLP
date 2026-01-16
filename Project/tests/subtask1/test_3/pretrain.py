from collections import Counter
from datasets import load_dataset

from datasets import load_dataset, concatenate_datasets

configs = [
    "stance_abortion",
    "stance_atheism",
    "stance_climate",
    "stance_feminist",
    "stance_hillary",
]

datasets = [load_dataset("tweet_eval", cfg) for cfg in configs]

train_ds = concatenate_datasets([d["train"] for d in datasets])
val_ds   = concatenate_datasets([d["validation"] for d in datasets])

print(train_ds[0])

ds = {
    "train": train_ds,
    "validation": val_ds
}


labels = [ex["label"] for ex in ds["train"]]
print("Raw label distribution:", Counter(labels))


STANCE_MAP = {0: 0, 1: 1, 2: 1}

def map_labels(example):
    example["labels"] = STANCE_MAP[example["label"]]
    return example

train_ds = train_ds.map(map_labels)
val_ds   = val_ds.map(map_labels)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-large")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds   = val_ds.map(tokenize, batched=True)

keep_cols = ["input_ids", "attention_mask", "labels"]

train_ds = train_ds.remove_columns(
    [c for c in train_ds.column_names if c not in keep_cols]
)
val_ds = val_ds.remove_columns(
    [c for c in val_ds.column_names if c not in keep_cols]
)

train_ds.set_format("torch")
val_ds.set_format("torch")
from collections import Counter

print("Train labels:", Counter(train_ds["labels"]))
print("Val labels:", Counter(val_ds["labels"]))


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-large",
    num_labels=2
)
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="./stance_pretrain",
    eval_steps=1000,
    save_steps=1000,
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=100,
    save_total_limit=1,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
)

trainer.train()

encoder = trainer.model.base_model
encoder.save_pretrained("./stance_encoder")
tokenizer.save_pretrained("./stance_encoder")
