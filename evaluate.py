"""Evaluation for the credit-default classifier.

This module computes the metrics that
actually describe performance on the rare positive class — confusion matrix,
precision, recall, F1, specificity, balanced accuracy — plus ROC AUC and
average precision over all thresholds.
"""
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             average_precision_score)


def collect_probs_and_labels(model, loader, device):
    """One inference pass over `loader`.

    Returns (y_true, y_prob) as 1-D numpy arrays of equal length. y_prob holds
    sigmoid probabilities in [0, 1], not hard labels: the threshold-independent
    metrics and step 5's threshold search both need the continuous scores.
    Runs under eval mode and no_grad.
    """
    model.eval()
    labels, probs = [],[]
    with torch.no_grad():
        for X,y in loader:
            X = X.to(device)
            p = torch.sigmoid(model(X))
            probs.append(p.cpu().numpy())
            labels.append(y.numpy())

    return np.concatenate(labels), np.concatenate(probs)



def metrics_at_threshold(y_true, y_prob, threshold=0.5):
    """Binary metrics obtained by cutting `y_prob` at `threshold`.

    Returns a dict: threshold, accuracy, precision, recall, f1, specificity,
    balanced_accuracy, cm.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn+fp) if (tn+fp) else 0.0

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": (recall + specificity)/ 2,
        "cm": cm
    }


def threshold_free_metrics(y_true, y_prob):
    """Metrics that summarise every threshold at once.

    roc_auc: ranking quality overall. average_precision: area under the
    precision-recall curve, the more informative summary when positives are
    rare, since it ignores the large true-negative mass that inflates ROC AUC.
    Returns a dict with both.
    """
    return {
    "roc_auc": roc_auc_score(y_true, y_prob),
    "average_precision": average_precision_score(y_true, y_prob),
}


def naive_baseline(y_true):
    """The majority-class predictor: everyone is labelled 'no default'.

    Returns its accuracy and recall. Any model whose accuracy fails to beat this
    has learned nothing useful, and its recall of 0.0 is what accuracy hides.
    """
    y_pred = np.zeros_like(y_true)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "recall": recall
    }


def format_confusion_matrix(cm, classes=("no default", "default")):
    """Render a 2x2 confusion matrix as an aligned text block.

    sklearn's binary layout is [[TN, FP], [FN, TP]]: rows are the TRUE class,
    columns the PREDICTED one.
    """
    tn, fp, fn, tp = cm.ravel()
    return (
                f"                  predicted\n"
        f"                {classes[0]:>10}  {classes[1]:>8}\n"
        f"true {classes[0]:>10}  {tn:>10}  {fp:>8}\n"
        f"true {classes[1]:>10}  {fn:>10}  {tp:>8}\n"
        f"\n"
        f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}"
    )


if __name__ == "__main__":
    from dataset import prepare_data
    from model import build_model, build_criterion, build_optimizer
    from train import fit

    torch.manual_seed(0)
    data = prepare_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(data["n_features"], device)
    frac = data["frac_default"]
    criterion = build_criterion(n_neg=1 - frac, n_pos=frac, device=device)
    optimizer = build_optimizer(model)
    fit(model, data["train_loader"], data["val_loader"],
        criterion, optimizer, device, epochs=5)

    y_true, y_prob = collect_probs_and_labels(model, data["test_loader"], device)

    # one probability per test sample, all between 0 and 1
    print("samples:", len(y_true), "| prob range: %.3f to %.3f" % (y_prob.min(), y_prob.max()))
    assert len(y_true) == len(data["test_loader"].dataset)
    assert y_prob.min() >= 0.0 and y_prob.max() <= 1.0

    m = metrics_at_threshold(y_true, y_prob, threshold=0.5)
    tf = threshold_free_metrics(y_true, y_prob)
    base = naive_baseline(y_true)

    # confusion matrix covers every sample
    cm = m["cm"]
    print("\n" + format_confusion_matrix(cm))
    assert cm.sum() == len(y_true)

    print("\n  accuracy  %.3f   (naive: %.3f)" % (m["accuracy"], base["accuracy"]))
    print("  recall    %.3f   (naive: %.3f)" % (m["recall"], base["recall"]))
    print("  precision %.3f" % m["precision"])
    print("  F1        %.3f" % m["f1"])
    print("  ROC AUC   %.3f" % tf["roc_auc"])

    # the model catches real defaults, the naive one catches none
    assert base["recall"] == 0.0
    assert m["recall"] > 0.30, "Recall too low — model collapsed to majority class."

    print("\nStep 4 passed.")