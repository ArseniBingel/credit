import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from evaluate import metrics_at_threshold


DEFAULT_GRID = np.linspace(0.0, 1.0, 101)


def sweep_thresholds(y_true, y_prob, grid=DEFAULT_GRID):
    """Precision, recall and F1 at every threshold in `grid`.

    Returns a dict of numpy arrays: threshold, precision, recall, f1 — all the
    same length as `grid`.
    """
    f1s, precs, recs = [],[],[]
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    return {
        "threshold": grid,
        "precision": np.array(precs),
        "f1": np.array(f1s),
        "recall": np.array(recs)
    }

        


def best_threshold_by_f1(y_true, y_prob, grid=DEFAULT_GRID):
    """Threshold with the highest F1. Returns (threshold, f1).
    """
    results = sweep_thresholds(y_true, y_prob, grid)
    best_f1_index = int(np.argmax(results["f1"]))
    return grid[best_f1_index], results["f1"][best_f1_index]


def best_threshold_at_precision_floor(y_true, y_prob, min_precision=0.40,
                                      grid=DEFAULT_GRID):
    """Highest-recall threshold whose precision still meets `min_precision`.

    Returns (threshold, recall, precision). If no threshold clears the floor,
    fall back to the F1-optimal threshold.
    """
    met_floor = False
    results = sweep_thresholds(y_true, y_prob, grid)
    highest_recall = 0.0
    fb_t, _ = best_threshold_by_f1(y_true, y_prob, grid)
    i = int(np.argmin(np.abs(grid - fb_t)))
    highest_tuple = (results["threshold"][i], results["recall"][i], results["precision"][i])

    for t in range(len(results["threshold"])):
        if (results["recall"][t] > highest_recall and results["precision"][t] >= min_precision):
            highest_recall = results["recall"][t]
            highest_tuple = (results["threshold"][t], results["recall"][t], results["precision"][t])
            met_floor = True
    if not met_floor:
        print(f"Warning: no threshold reached precision -> {min_precision:.2f} "
              f"Falling back to the F1-optimal threshold {highest_tuple[0]:.2f}")

    
    return highest_tuple

def compare_operating_points(y_true, y_prob, thresholds, labels=None):
    """Metrics side by side for several thresholds.

    Returns a pandas DataFrame with one row per threshold and columns for
    accuracy, precision, recall, f1 and the four confusion-matrix cells
    (tn, fp, fn, tp)."""
    rows = []
    for i, t in enumerate(thresholds):
        m = metrics_at_threshold(y_true, y_prob,threshold=t)
        tn, fp, fn, tp = m["cm"].ravel()
        rows.append({
            "label": labels[i] if labels else f"t={t:.2f}",
            "threshold": t,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })
    return pd.DataFrame(rows)
    


def save_checkpoint(model, threshold, in_features, path="credit_model.pth"):
    """Persist weights and the chosen threshold together."""
    ckpt = {
        "threshold": float(threshold),
        "state_dict": model.state_dict(),
        "in_features": int(in_features)
    }
    torch.save(ckpt,path)
    return path

def load_checkpoint(path, model_builder, device):
    """Rebuild the model from `path` and return (model, threshold)."""
    ckpt = torch.load(path, map_location=device)
    model = model_builder(ckpt["in_features"], device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, ckpt["threshold"]


if __name__ == "__main__":
    from dataset import prepare_data
    from model import build_model, build_criterion, build_optimizer
    from train import fit
    from evaluate import collect_probs_and_labels, metrics_at_threshold

    torch.manual_seed(0)
    data = prepare_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(data["n_features"], device)
    frac = data["frac_default"]
    criterion = build_criterion(n_neg=1 - frac, n_pos=frac, device=device)
    optimizer = build_optimizer(model)
    fit(model, data["train_loader"], data["val_loader"],
        criterion, optimizer, device, epochs=5)

    # --- pick the threshold on validation ---
    val_true, val_prob = collect_probs_and_labels(model, data["val_loader"], device)
    t_best, val_f1 = best_threshold_by_f1(val_true, val_prob)
    t_floor, rec_floor, prec_floor = best_threshold_at_precision_floor(val_true, val_prob)

    print("chosen threshold: %.2f  (val F1 %.3f)" % (t_best, val_f1))
    print("precision-floor threshold: %.2f  (recall %.3f, precision %.3f)"
          % (t_floor, rec_floor, prec_floor))
    assert 0.0 <= t_best <= 1.0

    # --- apply to TEST once ---
    test_true, test_prob = collect_probs_and_labels(model, data["test_loader"], device)
    m_default = metrics_at_threshold(test_true, test_prob, 0.5)
    m_tuned = metrics_at_threshold(test_true, test_prob, t_best)

    print("\ntest recall:    %.3f -> %.3f" % (m_default["recall"], m_tuned["recall"]))
    print("test precision: %.3f -> %.3f" % (m_default["precision"], m_tuned["precision"]))

    table = compare_operating_points(
        test_true, test_prob, [0.5, t_best, t_floor],
        labels=["default 0.50", "F1-optimal", "precision floor"])
    print("\n" + table.to_string(index=False))

    # every test sample lands in exactly one confusion cell
    assert (table[["tn", "fp", "fn", "tp"]].sum(axis=1) == len(test_true)).all()

    # --- save and reload: weights and threshold both survive ---
    save_checkpoint(model, t_best, data["n_features"])
    reborn, t_loaded = load_checkpoint("credit_model.pth", build_model, device)
    _, reborn_prob = collect_probs_and_labels(reborn, data["test_loader"], device)

    print("\nreloaded threshold:", t_loaded)
    assert abs(t_loaded - t_best) < 1e-9
    assert np.allclose(reborn_prob, test_prob, atol=1e-6), \
        "Reloaded model predicts differently — state_dict didn't survive."

    print("\nStep 5 passed.")