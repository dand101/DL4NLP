import os
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import TrainingArguments, DataCollatorWithPadding, set_seed
from sklearn.model_selection import train_test_split
from pathlib import Path

from dataset import TextDataset
from model import get_model_and_tokenizer
from trainer import WeightedTrainer, compute_class_weights
from evaluate import make_compute_metrics

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------

def read_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def balanced_train_test_split(df, label_col, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    counts = df[label_col].value_counts()
    if len(counts) < 2:
        return train_test_split(df, test_size=test_size, random_state=seed)

    n_test = int(round(len(df) * test_size))
    n_each = n_test // 2

    df0 = df[df[label_col] == 0]
    df1 = df[df[label_col] == 1]

    n_each = min(n_each, len(df0), len(df1))
    if n_each == 0:
        return train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df[label_col]
        )

    test0_idx = rng.choice(df0.index.to_numpy(), size=n_each, replace=False)
    test1_idx = rng.choice(df1.index.to_numpy(), size=n_each, replace=False)
    test_idx = np.concatenate([test0_idx, test1_idx])

    test_df = df.loc[test_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df = df.drop(index=test_idx).reset_index(drop=True)

    return train_df, test_df


# --------------------------------------------------------------------------------------
# One training run
# --------------------------------------------------------------------------------------

def run_single_experiment(cfg, train_seed, aug_seed, df, aug_df, dev_df):
    run_name = f"seed{train_seed}_aug{aug_seed}"
    run_dir = Path(cfg["outputs_dir"]) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n================ RUN {run_name} ================\n")

    set_seed(train_seed)

    # ---- split
    train_df, test_df = balanced_train_test_split(
        df,
        label_col=cfg["label_col"],
        test_size=cfg["test_size"],
        seed=train_seed,
    )

    # ---- augmentation
    label_col = cfg["label_col"]
    counts = train_df[label_col].value_counts()
    majority = counts.idxmax()
    minority = counts.idxmin()

    to_add = counts.max() - counts.min()
    aug_minority = aug_df[aug_df[label_col] == minority]

    if len(aug_minority) == 0:
        raise ValueError("No augmented samples for minority class")

    aug_minority = aug_minority.sample(
        n=min(to_add, len(aug_minority)),
        random_state=aug_seed,
    )

    train_df = pd.concat([train_df, aug_minority], ignore_index=True)

    # ---- model
    model, tokenizer = get_model_and_tokenizer(
        cfg["model_name"], cfg["num_labels"]
    )

    collator = DataCollatorWithPadding(tokenizer)

    train_ds = TextDataset(
        train_df,
        tokenizer,
        text_col=cfg["text_col"],
        label_col=label_col,
        max_length=cfg["max_length"],
    )

    test_ds = TextDataset(
        test_df,
        tokenizer,
        text_col=cfg["text_col"],
        label_col=label_col,
        max_length=cfg["max_length"],
    )

    class_weights = None
    if cfg.get("use_class_weights", True):
        class_weights = compute_class_weights(
            train_df[label_col].values,
            num_labels=cfg["num_labels"],
        )

    args = TrainingArguments(
        output_dir=str(run_dir / "ckpt"),
        eval_strategy=cfg["eval_strategy"],
        save_strategy=cfg["save_strategy"],
        logging_steps=cfg["logging_steps"],
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=int(cfg["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        load_best_model_at_end=True,
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        fp16=bool(cfg["fp16"] and torch.cuda.is_available()),
        report_to="none",
        seed=train_seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(cfg["confusion_matrix_file"]),
        class_weights=class_weights,
    )

    trainer.train()

    # ---- predict dev
    dev_ds = TextDataset(
        dev_df,
        tokenizer,
        text_col=cfg["text_col"],
        label_col=None,
        max_length=cfg["max_length"],
    )

    logits = trainer.predict(dev_ds).predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    np.save(run_dir / "dev_probs.npy", probs)

    return probs


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------

def main():
    cfg = read_cfg()

    os.makedirs(cfg["outputs_dir"], exist_ok=True)

    # ---- load data
    train_path = PROJECT_ROOT / cfg["data_root"] / cfg["train_file"]
    dev_path = PROJECT_ROOT / cfg["data_root"] / cfg["dev_file"]

    df = pd.read_csv(train_path)
    df = df[[cfg["id_col"], cfg["text_col"], cfg["label_col"]]].dropna()
    df[cfg["label_col"]] = df[cfg["label_col"]].astype(int)
    df[cfg["text_col"]] = df[cfg["text_col"]].astype(str)

    aug_df = pd.read_csv("train_augmented_backtranslation.csv")

    dev_df = pd.read_csv(dev_path)
    dev_df = dev_df[[cfg["id_col"], cfg["text_col"]]].dropna()
    dev_df[cfg["text_col"]] = dev_df[cfg["text_col"]].astype(str)

    all_probs = []

    for train_seed in cfg["train_seeds"]:
        for aug_seed in cfg["aug_seeds"]:
            probs = run_single_experiment(
                cfg, train_seed, aug_seed, df, aug_df, dev_df
            )
            all_probs.append(probs)

    all_probs = np.stack(all_probs, axis=0)
    ensemble_probs = all_probs.mean(axis=0)
    ensemble_pred = ensemble_probs.argmax(axis=1)

    np.save(Path(cfg["outputs_dir"]) / "ensemble_probs.npy", ensemble_probs)

    out = pd.DataFrame({
        cfg["id_col"]: dev_df[cfg["id_col"]].values,
        "pred_polarization": ensemble_pred,
    })

    out_path = Path(cfg["outputs_dir"]) / "ensemble_predictions.csv"
    out.to_csv(out_path, index=False)

    print("\n================ DONE ================\n")
    print("Saved ensemble predictions to:", out_path)


if __name__ == "__main__":
    main()
