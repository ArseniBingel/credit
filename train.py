"""Training and validation loop for the binary credit-default classifier.

Predictions for a single-logit head: sigmoid(logit) >= 0.5 -> class 1, else 0.
Because the loss carries pos_weight (step 2), the model predicts real positives
rather than collapsing to the majority class — step 4 measures whether those
predictions are actually good, since accuracy alone will be misleading here.
"""
import torch


def run_epoch(model, loader,device, train, optimizer=None,criterion=None):
    """One full pass over `loader`.

    train=True  -> train mode, gradients on, weights updated.
    train=False -> eval mode, gradients off, measurement only.
    Returns (avg_loss, accuracy). Loss is averaged per sample (weighted by batch
    size) so an odd-sized final batch does not skew the number.
    """
    model.train() if train else model.eval()
    grad = torch.enable_grad() if train else torch.no_grad()
    total_loss, correct, seen = 0.0, 0, 0
    with grad:
        for X,y in loader:
            X,y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits,y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds = (torch.sigmoid(logits) >= 0.5).int()
            correct += (preds == y.int()).sum().item()
            seen += len(y)
    return total_loss/seen, correct/seen


def fit(model, train_loader, val_loader, criterion, optimizer, device, epochs=5):
    """Train for `epochs`, evaluating on the validation set after each one.

    Returns a history dict with per-epoch lists: train_loss, train_acc,
    val_loss, val_acc. Prints one summary line per epoch.
    """
    history = {"train_loss": [], "train_acc":[], "val_loss":[], "val_acc":[]}

    for e in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion=criterion, optimizer=optimizer, device=device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader,criterion=criterion, device=device, train=False)
        print(f"train_loss: {train_loss}, train_acc: {train_acc}\n val_loss: {val_loss}, val_acc: {val_acc}")
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
    return history


if __name__ == "__main__":
    from dataset import prepare_data
    from model import build_model, build_criterion, build_optimizer

    data = prepare_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(data["n_features"], device)
    frac = data["frac_default"]
    criterion = build_criterion(n_neg=1 - frac, n_pos=frac, device=device)
    optimizer = build_optimizer(model)

    epochs = 5
    history = fit(model, data["train_loader"], data["val_loader"],
                  criterion, optimizer, device, epochs=epochs)

    # history is a dict with the four expected keys
    print("history keys:", list(history.keys()))
    assert set(history.keys()) == {"train_loss", "train_acc", "val_loss", "val_acc"}

    # each key holds one number PER EPOCH (so 5 numbers, not 1 — proves no overwrite)
    print("points per metric:", len(history["train_loss"]))
    assert len(history["train_loss"]) == epochs

    # training loss should DROP from first epoch to last (the model is learning)
    print("train loss: first %.4f -> last %.4f" %
          (history["train_loss"][0], history["train_loss"][-1]))
    assert history["train_loss"][-1] < history["train_loss"][0]

    print("val acc:    first %.4f -> last %.4f" %
          (history["val_acc"][0], history["val_acc"][-1]))

    print("fit check passed.")