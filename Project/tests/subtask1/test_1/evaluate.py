
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    report = classification_report(
        labels,
        preds,
        labels=[0, 1],
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm)
    print(cm)
    disp.plot()
    plt.savefig("outputs/confusion_matrix_xlm_r.png")
    plt.close()

    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_class_0": report["0"]["f1-score"],
        "f1_class_1": report["1"]["f1-score"],
        "precision_class_0": report["0"]["precision"],
        "recall_class_0": report["0"]["recall"],
        "precision_class_1": report["1"]["precision"],
        "recall_class_1": report["1"]["recall"],
    }
