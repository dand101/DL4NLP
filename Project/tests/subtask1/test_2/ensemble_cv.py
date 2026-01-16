import os
import yaml
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from transformers import TrainingArguments, DataCollatorWithPadding, set_seed

from dataset import TextDataset
from model import get_model_and_tokenizer
from trainer import WeightedTrainer, compute_class_weights
from evaluate import make_compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def read_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cast_cfg(cfg):
    cfg["learning_rate"] = float(cfg["learning_rate"])
    cfg["weight_decay"] = float(cfg["weight_decay"])
    cfg["warmup_ratio"] = float(cfg["warmup_ratio"])
    cfg["num_train_epochs"] = int(cfg["num_train_epochs"])
    cfg["per_device_train_batch_size"] = int(cfg["per_device_train_batch_size"])
    cfg["per_device_eval_batch_size"] = int(cfg["per_device_eval_batch_size"])
    cfg["gradient_accumulation_steps"] = int(cfg.get("gradient_accumulation_steps", 1))
    cfg["max_grad_norm"] = float(cfg.get("max_grad_norm", 1.0))
    cfg["label_smoothing_factor"] = float(cfg.get("label_smoothing_factor", 0.0))
    cfg["eval_steps"] = int(cfg.get("eval_steps", 100))
    cfg["save_steps"] = int(cfg.get("save_steps", 100))
    cfg["save_total_limit"] = int(cfg.get("save_total_limit", 1))
    cfg["cv_folds"] = int(cfg.get("cv_folds", 5))
    cfg["cv_start_fold"] = int(cfg.get("cv_start_fold", 1))
    cfg["cv_end_fold"] = int(cfg.get("cv_end_fold", cfg["cv_folds"]))
    cfg["seed"] = int(cfg["seed"])
    cfg["resume_from_checkpoint"] = bool(cfg.get("resume_from_checkpoint", False))
    return cfg


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def make_args(cfg, output_dir):
    return TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_steps"] if cfg["eval_strategy"] == "steps" else None,
        save_strategy=cfg["save_strategy"],
        save_steps=cfg["save_steps"] if cfg["save_strategy"] == "steps" else None,
        save_total_limit=cfg["save_total_limit"],
        logging_strategy="steps",
        logging_steps=int(cfg["logging_steps"]),
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        optim=cfg.get("optim", "adamw_torch"),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        label_smoothing_factor=cfg.get("label_smoothing_factor", 0.0),
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        load_best_model_at_end=bool(cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_macro_f1"),
        greater_is_better=bool(cfg.get("greater_is_better", True)),
        fp16=bool(cfg.get("fp16", False) and torch.cuda.is_available()),
        report_to=cfg.get("report_to", "none"),
        seed=cfg["seed"],
    )


def main():
    cfg = cast_cfg(read_cfg())
    set_seed(cfg["seed"])

    os.makedirs(cfg["outputs_dir"], exist_ok=True)

    train_path = PROJECT_ROOT / cfg["data_root"] / cfg["train_file"]
    dev_path = PROJECT_ROOT / cfg["data_root"] / cfg["dev_file"]

    df = pd.read_csv(train_path)
    df = df[[cfg["id_col"], cfg["text_col"], cfg["label_col"]]]
    df = df.dropna(subset=[cfg["text_col"], cfg["label_col"]])

    id_series = df[cfg["id_col"]]
    missing_id = id_series.isna() | id_series.astype(str).str.strip().eq("")
    df.loc[missing_id, cfg["id_col"]] = [f"aug_{i}" for i in df.index[missing_id]]

    df[cfg["label_col"]] = df[cfg["label_col"]].astype(int)
    df[cfg["text_col"]] = df[cfg["text_col"]].astype(str)
    df[cfg["id_col"]] = df[cfg["id_col"]].astype(str)

    dev_df = pd.read_csv(dev_path)
    dev_df = dev_df[[cfg["id_col"], cfg["text_col"]]]
    dev_df = dev_df.dropna(subset=[cfg["text_col"]])

    dev_missing_id = dev_df[cfg["id_col"]].isna() | dev_df[cfg["id_col"]].astype(str).str.strip().eq("")
    dev_df.loc[dev_missing_id, cfg["id_col"]] = [f"dev_{i}" for i in dev_df.index[dev_missing_id]]

    dev_df[cfg["text_col"]] = dev_df[cfg["text_col"]].astype(str)
    dev_df[cfg["id_col"]] = dev_df[cfg["id_col"]].astype(str)

    skf = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["seed"])

    fold_scores = []
    fold_metrics_rows = []
    dev_probs_sum = np.zeros((len(dev_df), cfg["num_labels"]), dtype=np.float32)
    completed_folds = 0

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, df[cfg["label_col"]].values), start=1):
        if fold < cfg["cv_start_fold"] or fold > cfg["cv_end_fold"]:
            continue

        fold_seed = cfg["seed"] + fold
        set_seed(fold_seed)

        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)

        fold_out = Path(cfg["outputs_dir"]) / f"cv_fold_{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)

        model, tokenizer = get_model_and_tokenizer(cfg["model_name"], cfg["num_labels"])
        collator = DataCollatorWithPadding(tokenizer=tokenizer)

        train_ds = TextDataset(
            train_df,
            tokenizer,
            text_col=cfg["text_col"],
            label_col=cfg["label_col"],
            max_length=cfg["max_length"],
        )
        val_ds = TextDataset(
            val_df,
            tokenizer,
            text_col=cfg["text_col"],
            label_col=cfg["label_col"],
            max_length=cfg["max_length"],
        )

        class_weights = None
        if cfg.get("use_class_weights", True):
            class_weights = compute_class_weights(
                train_df[cfg["label_col"]].values,
                num_labels=cfg["num_labels"],
            )

        cm_path = str(Path(cfg["outputs_dir"]) / f"confusion_matrix_fold_{fold}.png")
        compute_metrics = make_compute_metrics(cm_path)

        args = make_args(cfg, fold_out)

        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
        )

        if cfg.get("resume_from_checkpoint", False):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

        metrics = trainer.evaluate()
        f1 = float(metrics.get("eval_macro_f1", np.nan))
        fold_scores.append(f1)

        row = {"fold": fold, "seed": fold_seed}
        row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float, np.number))})
        fold_metrics_rows.append(row)

        print(f"\n[FOLD {fold}] eval_macro_f1 = {f1}")

        dev_ds = TextDataset(
            dev_df,
            tokenizer,
            text_col=cfg["text_col"],
            label_col=None,
            max_length=cfg["max_length"],
        )
        dev_logits = trainer.predict(dev_ds).predictions
        dev_probs = softmax(dev_logits, axis=-1)
        dev_probs_sum += dev_probs
        completed_folds += 1

    if completed_folds == 0:
        raise RuntimeError("No folds were run. Check cv_start_fold/cv_end_fold/cv_folds in config.yaml.")

    dev_probs_avg = dev_probs_sum / float(completed_folds)
    dev_pred = np.argmax(dev_probs_avg, axis=-1)

    mean_f1 = float(np.nanmean(fold_scores))
    std_f1 = float(np.nanstd(fold_scores))
    print("\n=== CV SUMMARY ===")
    print("Folds run:", list(range(cfg["cv_start_fold"], cfg["cv_end_fold"] + 1)))
    print("Fold macro-F1:", fold_scores)
    print(f"Mean macro-F1: {mean_f1:.6f} | Std: {std_f1:.6f} | Completed folds: {completed_folds}")

    metrics_path = Path(cfg["outputs_dir"]) / "cv_fold_metrics.csv"
    pd.DataFrame(fold_metrics_rows).to_csv(metrics_path, index=False)
    print("Saved:", metrics_path)

    out_path = Path(cfg["outputs_dir"]) / "eng_dev_predictions_cv_ensemble.csv"
    out = pd.DataFrame({cfg["id_col"]: dev_df[cfg["id_col"]].values, "pred_polarization": dev_pred})
    out.to_csv(out_path, index=False)
    print("Saved dev predictions to:", out_path)


if __name__ == "__main__":
    main()
