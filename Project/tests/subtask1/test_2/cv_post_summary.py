import json
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from typing import Optional, Dict, Any, Tuple

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    classification_report,
    confusion_matrix,
)


# -------------------------
# Config / IO helpers
# -------------------------
def read_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(p: Path):
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def best_eval_from_trainer_state(trainer_state: dict):
    """
    Pick the best eval row from trainer_state["log_history"].
    Prefer best eval_macro_f1; otherwise last eval row.
    """
    hist = trainer_state.get("log_history", [])
    eval_rows = [
        r for r in hist
        if isinstance(r, dict) and ("eval_loss" in r or any(k.startswith("eval_") for k in r))
    ]
    if not eval_rows:
        return None

    if any("eval_macro_f1" in r for r in eval_rows):
        rows = [
            r for r in eval_rows
            if "eval_macro_f1" in r and isinstance(r["eval_macro_f1"], (int, float))
        ]
        if rows:
            return max(rows, key=lambda r: r["eval_macro_f1"])

    return eval_rows[-1]


def extract_metrics_from_fold_dir(fold_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Returns a dict of eval metrics from fold_dir, or None if nothing found.
    """
    all_results = load_json(fold_dir / "all_results.json")
    if isinstance(all_results, dict) and any(k.startswith("eval_") for k in all_results.keys()):
        return all_results

    ts = load_json(fold_dir / "trainer_state.json")
    if isinstance(ts, dict):
        best = best_eval_from_trainer_state(ts)
        if best:
            return best

    ckpts = sorted([p for p in fold_dir.glob("checkpoint-*") if p.is_dir()])
    if ckpts:
        def ckpt_step(p):
            try:
                return int(p.name.split("-")[-1])
            except Exception:
                return -1

        ckpts = sorted(ckpts, key=ckpt_step)
        ts2 = load_json(ckpts[-1] / "trainer_state.json")
        if isinstance(ts2, dict):
            best = best_eval_from_trainer_state(ts2)
            if best:
                return best

    return None


# -------------------------
# Prediction loading (for ensemble inference metrics)
# -------------------------
def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def try_load_fold_predictions(fold_dir: Path) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Try to load per-example model outputs for the eval split of a fold.

    Supported formats (any one is fine):
      - eval_predictions.npz containing logits and labels
      - predictions.npz containing logits and labels
      - eval_logits.npy + eval_labels.npy
      - logits.npy + labels.npy

    Returns:
      (logits, labels) where labels may be None if not saved.
    """
    for name in ["eval_predictions.npz", "predictions.npz"]:
        p = fold_dir / name
        if p.exists():
            data = np.load(p, allow_pickle=True)
            logits = None
            labels = None
            for k in ["logits", "preds", "predictions"]:
                if k in data:
                    logits = data[k]
                    break
            for k in ["labels", "y", "targets"]:
                if k in data:
                    labels = data[k]
                    break
            if logits is not None:
                return logits, labels

    candidates = [
        (fold_dir / "eval_logits.npy", fold_dir / "eval_labels.npy"),
        (fold_dir / "logits.npy", fold_dir / "labels.npy"),
    ]
    for lp, yp in candidates:
        if lp.exists():
            logits = np.load(lp)
            labels = np.load(yp) if yp.exists() else None
            return logits, labels

    ckpts = sorted([p for p in fold_dir.glob("checkpoint-*") if p.is_dir()])
    if ckpts:
        def ckpt_step(p):
            try:
                return int(p.name.split("-")[-1])
            except Exception:
                return -1
        ckpts = sorted(ckpts, key=ckpt_step)
        last = ckpts[-1]
        for name in ["eval_predictions.npz", "predictions.npz"]:
            p = last / name
            if p.exists():
                data = np.load(p, allow_pickle=True)
                logits = None
                labels = None
                for k in ["logits", "preds", "predictions"]:
                    if k in data:
                        logits = data[k]
                        break
                for k in ["labels", "y", "targets"]:
                    if k in data:
                        labels = data[k]
                        break
                if logits is not None:
                    return logits, labels

        for lp_name, yp_name in [("eval_logits.npy", "eval_labels.npy"), ("logits.npy", "labels.npy")]:
            lp = last / lp_name
            yp = last / yp_name
            if lp.exists():
                logits = np.load(lp)
                labels = np.load(yp) if yp.exists() else None
                return logits, labels

    return None


def compute_ensemble_metrics(fold_dirs: list[Path]) -> Optional[Dict[str, Any]]:
    """
    Compute metrics from an ensemble by averaging probabilities across folds.
    Requires per-example logits for the eval split saved in each fold dir.

    Returns dict with accuracy, macro_f1, macro_precision, macro_recall, per-class P/R/F1, confusion matrix.
    """
    fold_logits = []
    y_true = None

    for fd in fold_dirs:
        loaded = try_load_fold_predictions(fd)
        if loaded is None:
            return None
        logits, labels = loaded
        fold_logits.append(logits)

        if labels is not None:
            if y_true is None:
                y_true = labels
            else:
                pass

    if y_true is None:
        return None

    probs = np.mean([softmax(l) for l in fold_logits], axis=0)
    y_pred = np.argmax(probs, axis=-1)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    macro_p, macro_r, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    p_cls, r_cls, f_cls, supp = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "ensemble_accuracy": acc,
        "ensemble_macro_f1": macro_f1,
        "ensemble_macro_precision": float(macro_p),
        "ensemble_macro_recall": float(macro_r),
        "precision_0": float(p_cls[0]),
        "recall_0": float(r_cls[0]),
        "f1_0": float(f_cls[0]),
        "support_0": int(supp[0]),
        "precision_1": float(p_cls[1]),
        "recall_1": float(r_cls[1]),
        "f1_1": float(f_cls[1]),
        "support_1": int(supp[1]),
        "confusion_matrix": cm.tolist(),
    }


# -------------------------
# Main
# -------------------------
def main():
    cfg = read_cfg()
    out_dir = Path(cfg.get("outputs_dir", "outputs"))

    fold_dirs = sorted([p for p in out_dir.glob("cv_fold_*") if p.is_dir()])
    if not fold_dirs:
        raise FileNotFoundError(f"No cv_fold_* directories found under: {out_dir}")

    rows = []
    for fd in fold_dirs:
        try:
            fold_num = int(fd.name.split("_")[-1])
        except Exception:
            fold_num = fd.name

        m = extract_metrics_from_fold_dir(fd)
        if m is None:
            print(f"[SKIP] {fd} (no metrics found)")
            continue

        clean = {k: v for k, v in m.items() if isinstance(v, (int, float))}
        clean["fold"] = fold_num
        clean["fold_dir"] = str(fd)
        rows.append(clean)
        print(f"[OK] fold {fold_num} -> keys: {sorted(list(clean.keys()))}")

    if not rows:
        raise RuntimeError("Found cv_fold_* dirs but could not extract metrics from any of them.")

    df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)

    print("\n=== CV FOLD SUMMARY (from cv_fold_* folders) ===")
    print(df[[c for c in ["fold", "eval_macro_f1", "eval_accuracy", "eval_precision_0", "eval_precision_1"] if c in df.columns]].to_string(index=False))

    merged_path = out_dir / "cv_fold_metrics_merged.csv"
    df.to_csv(merged_path, index=False)
    print("\nSaved merged metrics to:", merged_path)

    ens = compute_ensemble_metrics(fold_dirs)
    if ens is None:
        print("\n=== ENSEMBLE INFERENCE METRICS ===")
        print("Could not compute ensemble metrics because per-example eval logits/labels were not found.")
        print("To enable this, save per-fold eval predictions (logits + labels) to one of these formats per fold:")
        print("  - eval_predictions.npz (keys: logits, labels) OR predictions.npz")
        print("  - eval_logits.npy + eval_labels.npy")
        print("  - logits.npy + labels.npy")
        print("\nNOTE: If you want an ensemble over the *full dataset* (OOF), you must save out-of-fold predictions")
        print("with original example indices, not just per-fold validation splits.")
        return

    print("\n=== ENSEMBLE INFERENCE METRICS (soft-vote avg probs) ===")
    print(f"Accuracy:        {ens['ensemble_accuracy']:.6f}")
    print(f"Macro Precision: {ens['ensemble_macro_precision']:.6f}")
    print(f"Macro Recall:    {ens['ensemble_macro_recall']:.6f}")
    print(f"Macro F1:        {ens['ensemble_macro_f1']:.6f}")
    print("\nPer-class:")
    print(f"  Class 0: P={ens['precision_0']:.6f} R={ens['recall_0']:.6f} F1={ens['f1_0']:.6f} (n={ens['support_0']})")
    print(f"  Class 1: P={ens['precision_1']:.6f} R={ens['recall_1']:.6f} F1={ens['f1_1']:.6f} (n={ens['support_1']})")
    print("\nConfusion matrix [ [tn fp], [fn tp] ]:")
    print(np.array(ens["confusion_matrix"]))

    ens_path = out_dir / "cv_ensemble_metrics.json"
    with ens_path.open("w", encoding="utf-8") as f:
        json.dump(ens, f, indent=2)
    print("\nSaved ensemble metrics to:", ens_path)


if __name__ == "__main__":
    main()
