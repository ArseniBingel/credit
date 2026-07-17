# step2_model.py
"""Model, loss, and optimizer for the credit-default classifier.

Binary classification with a single-logit head. The loss is BCEWithLogitsLoss
weighted by pos_weight to counter the ~22% / 78% class imbalance, so the model
is penalised more for missing the rare 'default' class than the common one.
"""
import torch
import torch.nn as nn


class CreditDefaultModel(nn.Module):
    """Feed-forward net: in_features -> hidden stack -> 1 logit.

    forward returns raw logits of shape (batch,). No sigmoid here — it is applied
    inside BCEWithLogitsLoss, which fuses sigmoid + BCE for numerical stability.
    The output is squeezed to (batch,) on purpose
    """

    def __init__(self, in_features, hidden=(64, 32), p_drop=0.3):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden:
            layers.append(nn.Linear(prev,h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_drop))
            prev = h
        self.feature_stack = nn.Sequential(*layers)
        self.head = nn.Linear(hidden[-1],1)

    def forward(self, x):
        x = self.feature_stack(x)
        return self.head(x).squeeze(-1)


def build_model(in_features, device):
    """Instantiate CreditDefaultModel and move it to `device`. Returns the model."""
    model = CreditDefaultModel(in_features)
    return model.to(device)
    


def build_criterion(n_neg, n_pos, device):
    """BCEWithLogitsLoss with pos_weight = n_neg / n_pos."""
    pos_weight = torch.tensor([n_neg/n_pos], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def build_optimizer(model, lr=1e-3, weight_decay=0.0):
    """Adam over the model's parameters. weight_decay adds mild L2 regularisation."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


if __name__ == "__main__":
    from dataset import prepare_data

    data = prepare_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build the model, run one batch through it
    model = build_model(data["n_features"], device)
    xb, yb = next(iter(data["train_loader"]))
    xb, yb = xb.to(device), yb.to(device)
    logits = model(xb)

    print("logits shape:", logits.shape)        # want (64,) — one number per sample
    assert logits.shape == (xb.shape[0],)

    print("num params:  ", sum(p.numel() for p in model.parameters()))

    # weighted loss for the imbalance
    frac = data["frac_default"]
    criterion = build_criterion(n_neg=1 - frac, n_pos=frac, device=device)
    print("pos_weight:  ", criterion.pos_weight.item())   # want > 1 (rare class up-weighted)
    assert criterion.pos_weight.item() > 1.0

    # loss on a real batch should be one finite number
    loss = criterion(logits, yb)
    print("loss:        ", loss.item())
    assert torch.isfinite(loss)

    optimizer = build_optimizer(model)

    print("Step 2 passed.")