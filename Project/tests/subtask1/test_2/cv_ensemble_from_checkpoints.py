"""
dev_ensemble_from_checkpoints.py

Soft-voting ensemble inference on DEV using cv_fold_* checkpoints.

- Loads config.yaml (same keys as your training scripts)
- Loads dev CSV (id + text)
- For each fold directory (outputs/cv_fold_k), loads the latest checkpoint
- Runs inference on DEV, gets probabilities, averages across folds (soft voting)
- Saves:
    outputs/dev_pred_ensemble.csv      (id, pred_polarization)
    outputs/dev_probs_ensemble.csv     (id, prob_0, prob_1)
    outputs/dev_ensemble_meta.json     (info about folds used, etc.)

IMPORTANT:
- This script does NOT compute accuracy/precision/recall/F1 on DEV unless DEV has labels.
  In typical shared-task setups DEV is unlabeled; metrics are computed on the hidden server.
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def read_cfg(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits)
    return e / np.sum(e, axis=axis, keepdims=True)


def find_latest_checkpoint(fold_dir: Path) -> Path:
    """
    Pick checkpoint-* with the largest step number under fold_dir.
    Falls back to fold_dir if no checkpoint-* exists.
    """
    ckpts = [p for p in fold_dir.glob("checkpoint-*") if p.is_dir()]
    if not ckpts:
        return fold_dir

    def step_num(p: Path) -> int:
        try:
            return int(p.name.split("-")[-1])
        except Exception:
            return -1

    ckpts.sort(key=step_num)
    return ckpts[-1]


@torch.inference_mode()
def predict_probs(
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        texts: list[str],
        max_length: int,
        batch_size: int,
        device: torch.device,
) -> np.ndarray:
    if batch_size is None or batch_size <= 0:
        raise ValueError(
            f"Invalid batch_size={batch_size}. "
            f"Check per_device_eval_batch_size in config.yaml."
        )

    if texts is None or len(texts) == 0:
        raise ValueError(
            "Got 0 texts to predict on (texts list is empty). "
            "Check dev loading / column names / dropna."
        )

    model.eval()
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.detach().float().cpu().numpy()
        all_probs.append(softmax_np(logits, axis=-1))

    return np.concatenate(all_probs, axis=0)


def main():
    cfg = read_cfg()

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    outputs_dir = Path(cfg.get("outputs_dir", "outputs"))
    data_root = PROJECT_ROOT / cfg["data_root"]
    dev_path = data_root / cfg["dev_file"]

    id_col = cfg["id_col"]
    text_col = cfg["text_col"]

    num_labels = int(cfg["num_labels"])
    max_length = int(cfg["max_length"])
    k_folds = int(cfg.get("cv_folds", 5))
    seed = int(cfg.get("seed", 42))
    batch_size = int(cfg.get("per_device_eval_batch_size", 8))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -------------------------
    # Load DEV
    # -------------------------
    print("DEV path:", dev_path)
    dev_df = pd.read_csv(dev_path)
    dev_df.columns = [c.strip() for c in dev_df.columns]

    print("DEV columns:", dev_df.columns.tolist())
    print("DEV rows before dropna:", len(dev_df))

    missing = [c for c in [id_col, text_col] if c not in dev_df.columns]
    if missing:
        raise KeyError(
            f"DEV is missing columns {missing}. "
            f"Found columns: {dev_df.columns.tolist()} "
            f"(Check id_col/text_col in config.yaml)."
        )

    dev_df = dev_df.dropna(subset=[id_col, text_col]).reset_index(drop=True)
    print("DEV rows after dropna:", len(dev_df))
    if len(dev_df) == 0:
        raise RuntimeError(
            "DEV became empty after dropna. "
            "Either the text column name is wrong or all texts are missing."
        )

    dev_df[text_col] = dev_df[text_col].astype(str)
    dev_texts = dev_df[text_col].tolist()

    print("Sample DEV text:", dev_texts[0][:200])

    # -------------------------
    # Soft voting across folds
    # -------------------------
    probs_sum = np.zeros((len(dev_df), num_labels), dtype=np.float32)
    used_folds = 0

    for fold in range(1, k_folds + 1):
        fold_dir = outputs_dir / f"cv_fold_{fold}"
        if not fold_dir.exists():
            print(f"[SKIP] fold {fold} missing dir: {fold_dir}")
            continue

        ckpt_dir = find_latest_checkpoint(fold_dir)
        print(f"[FOLD {fold}] Loading checkpoint: {ckpt_dir}")

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
        model.to(device)

        probs = predict_probs(
            model=model,
            tokenizer=tokenizer,
            texts=dev_texts,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )

        if probs.shape != (len(dev_df), num_labels):
            raise RuntimeError(
                f"Bad probs shape from fold {fold}: {probs.shape}, "
                f"expected ({len(dev_df)}, {num_labels})"
            )

        probs_sum += probs.astype(np.float32)
        used_folds += 1

        del model
        torch.cuda.empty_cache()

    if used_folds == 0:
        raise RuntimeError(
            "No folds were used. Check outputs_dir/cv_fold_* folders and cv_folds in config.yaml."
        )

    probs_avg = probs_sum / float(used_folds)
    pred = np.argmax(probs_avg, axis=-1)

    # -------------------------
    # Save outputs
    # -------------------------
    outputs_dir.mkdir(parents=True, exist_ok=True)

    probs_out = outputs_dir / "dev_probs_ensemble.csv"
    pd.DataFrame(
        {
            id_col: dev_df[id_col].values,
            "prob_0": probs_avg[:, 0],
            "prob_1": probs_avg[:, 1],
        }
    ).to_csv(probs_out, index=False)
    print("Saved:", probs_out)

    pred_out = outputs_dir / "dev_pred_ensemble.csv"
    pd.DataFrame(
        {
            id_col: dev_df[id_col].values,
            "pred_polarization": pred,
        }
    ).to_csv(pred_out, index=False)
    print("Saved:", pred_out)

    meta_out = outputs_dir / "dev_ensemble_meta.json"
    meta = {
        "used_folds": used_folds,
        "k_folds_cfg": k_folds,
        "seed": seed,
        "num_labels": num_labels,
        "max_length": max_length,
        "batch_size": batch_size,
        "outputs_dir": str(outputs_dir),
        "dev_path": str(dev_path),
    }
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved:", meta_out)

    print("\nNOTE: DEV metrics (accuracy/precision/recall/F1) require DEV labels.")
    print("If dev has no 'polarization' column, you cannot compute metrics locally; use leaderboard.")


if __name__ == "__main__":
    main()
