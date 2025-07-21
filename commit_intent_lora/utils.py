from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

INTENT_LABELS = ["bug", "refactor", "deprecation", "feature", "merge", "resource", "test"]
label2id = {label: i for i, label in enumerate(INTENT_LABELS)}
id2label = {i: label for label, i in label2id.items()}
NUM_LABELS = len(INTENT_LABELS)

def compute_metrics(p):
    logits = p.predictions
    labels = p.label_ids

    # 1. Apply sigmoid to logits
    probs = 1 / (1 + np.exp(-logits))

    print("=== [Debug Info] ===")
    print("Logits stats → mean:", logits.mean(), "std:", logits.std())
    print("Sigmoid mean:", probs.mean())
    print("Gold label counts:", labels.sum(axis=0).astype(int).tolist())
    print("=====================")

    results = {}
    for threshold in [0.5, 0.4, 0.3]:
        preds = (probs > threshold).astype(int)

        print(f"\n--- Threshold = {threshold} ---")
        print("Predicted label counts:", preds.sum(axis=0).astype(int).tolist())

        results.update({
            f"micro/f1@{threshold}": f1_score(labels, preds, average="micro", zero_division=0),
            f"macro/f1@{threshold}": f1_score(labels, preds, average="macro", zero_division=0),
            f"micro/precision@{threshold}": precision_score(labels, preds, average="micro", zero_division=0),
            f"micro/recall@{threshold}": recall_score(labels, preds, average="micro", zero_division=0),
        })

    return results
