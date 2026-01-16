import os
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import TrainingArguments, DataCollatorWithPadding, set_seed
from sklearn.model_selection import train_test_split

from dataset import TextDataset
from model import get_model_and_tokenizer
from trainer import WeightedTrainer, compute_class_weights
from evaluate import make_compute_metrics
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
        return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[label_col])

    test0_idx = rng.choice(df0.index.to_numpy(), size=n_each, replace=False)
    test1_idx = rng.choice(df1.index.to_numpy(), size=n_each, replace=False)
    test_idx = np.concatenate([test0_idx, test1_idx])

    test_df = df.loc[test_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df = df.drop(index=test_idx).reset_index(drop=True)
    return train_df, test_df


def read_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = read_cfg()

    cfg["learning_rate"] = float(cfg["learning_rate"])
    cfg["weight_decay"] = float(cfg["weight_decay"])
    cfg["warmup_ratio"] = float(cfg["warmup_ratio"])
    cfg["num_train_epochs"] = int(cfg["num_train_epochs"])
    cfg["per_device_train_batch_size"] = int(cfg["per_device_train_batch_size"])
    cfg["per_device_eval_batch_size"] = int(cfg["per_device_eval_batch_size"])
    cfg["gradient_accumulation_steps"] = int(cfg.get("gradient_accumulation_steps", 1))

    set_seed(cfg["seed"])
    os.makedirs(cfg["outputs_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    train_path = PROJECT_ROOT / cfg["data_root"] / cfg["train_file"]
    dev_path = PROJECT_ROOT / cfg["data_root"] / cfg["dev_file"]
    train_path = str(train_path)
    dev_path = str(dev_path)

    import pandas as pd

    SEED = 42
    N_PER_CLASS = 20_000

    # df = pd.read_parquet(
    #     "hf://datasets/valurank/hate-multi/data/train-00000-of-00001.parquet"
    # )
    #
    # # Check label distribution (always do this once)
    # print(df["label"].value_counts())
    #
    # # Sample N_PER_CLASS from each label
    # df = (
    #     df.groupby("label", group_keys=False)
    #     .apply(lambda x: x.sample(
    #         n=min(len(x), N_PER_CLASS),
    #         random_state=SEED
    #     ))
    #     .reset_index(drop=True)
    # )
    #
    # print(df["label"].value_counts())
    # print("Total samples:", len(df))

    df = pd.read_csv(train_path)

    df[cfg["label_col"]] = df[cfg["label_col"]].astype(int)
    df[cfg["text_col"]] = df[cfg["text_col"]].astype(str)

    if cfg.get("balance_test", True):
        train_df, test_df = balanced_train_test_split(
            df, label_col=cfg["label_col"], test_size=cfg["test_size"], seed=cfg["seed"]
        )
    else:
        train_df, test_df = train_test_split(
            df, test_size=cfg["test_size"], random_state=cfg["seed"], stratify=df[cfg["label_col"]]
        )

    print("Train label counts:\n", train_df[cfg["label_col"]].value_counts())
    print("Test label counts:\n", test_df[cfg["label_col"]].value_counts())

    model, tokenizer = get_model_and_tokenizer(cfg["model_name"], cfg["num_labels"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_ds = TextDataset(
        train_df, tokenizer,
        text_col=cfg["text_col"],
        label_col=cfg["label_col"],
        max_length=cfg["max_length"],
    )
    test_ds = TextDataset(
        test_df, tokenizer,
        text_col=cfg["text_col"],
        label_col=cfg["label_col"],
        max_length=cfg["max_length"],
    )

    class_weights = None
    if cfg.get("use_class_weights", True):
        class_weights = compute_class_weights(train_df[cfg["label_col"]].values, num_labels=cfg["num_labels"])

    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg.get("eval_steps", None),

        save_strategy=cfg["save_strategy"],
        save_steps=cfg.get("save_steps", None),

        save_total_limit=cfg.get("save_total_limit", 1),

        logging_strategy="steps",
        logging_steps=cfg["logging_steps"],
        learning_rate=cfg["learning_rate"],

        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        optim=cfg.get("optim", "adamw_torch_fused"),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),

        label_smoothing_factor=cfg.get("label_smoothing_factor", 0.0),

        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        num_train_epochs=cfg["num_train_epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        fp16=bool(cfg["fp16"] and torch.cuda.is_available()),
        report_to=cfg["report_to"],
        seed=cfg["seed"],
    )

    compute_metrics = make_compute_metrics(cfg["confusion_matrix_file"])

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    print("Num train epochs:", args.num_train_epochs)
    print("Total training steps:", trainer.state.max_steps)

    metrics = trainer.evaluate()
    pred_output = trainer.predict(test_ds)
    logits = pred_output.predictions
    y_pred = np.argmax(logits, axis=1)
    y_true = test_df[cfg["label_col"]].values
    analysis_df = test_df.copy().reset_index(drop=True)
    analysis_df["y_true"] = y_true
    analysis_df["y_pred"] = y_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    analysis_df["conf_pred"] = probs.max(axis=1)
    false_positives = analysis_df[
        (analysis_df["y_true"] == 0) & (analysis_df["y_pred"] == 1)
        ]

    false_negatives = analysis_df[
        (analysis_df["y_true"] == 1) & (analysis_df["y_pred"] == 0)
        ]
    fp_path = os.path.join(cfg["outputs_dir"], "false_positives.csv")
    fn_path = os.path.join(cfg["outputs_dir"], "false_negatives.csv")

    false_positives.to_csv(fp_path, index=False)
    false_negatives.to_csv(fn_path, index=False)

    print(f"Saved FP to: {fp_path}")
    print(f"Saved FN to: {fn_path}")

    print("\nEval metrics:", metrics)

    log_path = os.path.join(cfg["outputs_dir"], "training_log.csv")
    hist = pd.DataFrame(trainer.state.log_history)
    hist.to_csv(log_path, index=False)
    print("Saved training log to:", log_path)

    train_loss = hist[hist.get("loss").notna()][["step", "loss"]].dropna()
    eval_metrics = hist[hist.get("eval_loss").notna()].copy()

    if len(train_loss) > 0:
        plt.figure()
        plt.plot(train_loss["step"], train_loss["loss"])
        plt.title("Training loss over steps")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.tight_layout()
        outp = os.path.join(cfg["outputs_dir"], "curve_train_loss.png")
        plt.savefig(outp)
        plt.close()
        print("Saved:", outp)

    if len(eval_metrics) > 0:
        if "eval_macro_f1" in eval_metrics.columns:
            plt.figure()
            plt.plot(eval_metrics["step"], eval_metrics["eval_macro_f1"])
            plt.title("Eval Macro-F1 over steps")
            plt.xlabel("Step")
            plt.ylabel("Macro-F1")
            plt.tight_layout()
            outp = os.path.join(cfg["outputs_dir"], "curve_eval_macro_f1.png")
            plt.savefig(outp)
            plt.close()
            print("Saved:", outp)

        plt.figure()
        plt.plot(eval_metrics["step"], eval_metrics["eval_loss"])
        plt.title("Eval loss over steps")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.tight_layout()
        outp = os.path.join(cfg["outputs_dir"], "curve_eval_loss.png")
        plt.savefig(outp)
        plt.close()
        print("Saved:", outp)

    ####################################################################################################################################
    dev_df = pd.read_csv(dev_path)
    dev_df = dev_df[[cfg["id_col"], cfg["text_col"]]].dropna()
    dev_df[cfg["text_col"]] = dev_df[cfg["text_col"]].astype(str)

    dev_ds = TextDataset(
        dev_df, tokenizer,
        text_col=cfg["text_col"],
        label_col=None,
        max_length=cfg["max_length"],
    )

    dev_logits = trainer.predict(dev_ds).predictions
    dev_pred = np.argmax(dev_logits, axis=-1)

    out = pd.DataFrame({cfg["id_col"]: dev_df[cfg["id_col"]].values, "pred_polarization": dev_pred})
    out.to_csv(cfg["predictions_file"], index=False)
    print("Saved dev predictions to:", cfg["predictions_file"])


if __name__ == "__main__":
    main()
