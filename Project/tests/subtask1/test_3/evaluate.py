import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

def make_compute_metrics(confusion_matrix_path: str, print_in_terminal: bool = True):
    os.makedirs(os.path.dirname(confusion_matrix_path), exist_ok=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")

        pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1], average=None)
        cm = confusion_matrix(labels, preds, labels=[0, 1])

        if print_in_terminal:
            print("\n=== Confusion Matrix (rows=true, cols=pred) ===")
            print(cm)
            print("\n=== Classification report ===")
            print(classification_report(labels, preds, digits=4))

        plt.figure(figsize=(4, 3))
        plt.imshow(cm)
        plt.xticks([0, 1], ["0", "1"])
        plt.yticks([0, 1], ["0", "1"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(confusion_matrix_path)
        plt.close()

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "precision_0": pr[0],
            "recall_0": rc[0],
            "f1_0": f1[0],
            "precision_1": pr[1],
            "recall_1": rc[1],
            "f1_1": f1[1],
        }

    return compute_metrics
