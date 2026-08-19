# credit-default-classifier

A PyTorch binary classifier for credit card default, built around the problem
that accuracy is uninformative when 22% of the data is positive.

The pipeline is built end to end, measured at three operating points rather than
one, and the decision threshold is treated as a hyperparameter selected on
validation instead of left at 0.5.

## Results

### Default prediction (UCI id=350, 30,000 clients, 23 features, 22.1% positive)

Test split, 6,000 clients, 1,327 of them defaults.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Majority class (floor) | 0.779 | — | 0.000 | 0.000 |
| MLP, threshold 0.50 | 0.754 | 0.457 | 0.589 | 0.514 |
| MLP, threshold 0.61 (F1-optimal on val) | 0.793 | 0.534 | 0.497 | 0.515 |
| MLP, threshold 0.45 (precision floor 0.40) | 0.701 | 0.396 | 0.667 | 0.497 |

ROC AUC 0.762.

### Confusion matrices

Threshold 0.50:

|  | predicted no default | predicted default |
|---|---|---|
| **true no default** | 3745 | 928 |
| **true default** | 546 | 781 |

Threshold 0.61:

|  | predicted no default | predicted default |
|---|---|---|
| **true no default** | 4098 | 575 |
| **true default** | 668 | 659 |

## Findings

**The majority-class floor is 0.779 accuracy with zero recall.** Any accuracy
figure for this dataset has to be read against that number. The model at
threshold 0.50 scores 0.754, below the floor, while detecting 589 of the 1,327
actual defaults.

**Weighting the loss is what makes the minority class predictable at all.**
`pos_weight` is set to `n_neg / n_pos` = 3.52. Without it the model converges to
predicting no default for every client, which is the loss minimum the imbalance
rewards.

**Tuning the threshold moved the precision-recall balance but not F1.** Test F1
is 0.5145 at threshold 0.50 and 0.5146 at 0.61. What changed is the mix: precision rose from 0.457 to 0.534 and recall
fell from 0.589 to 0.497.

**Accuracy and F1 disagree about which threshold is better.** Threshold 0.61
scores 0.793 accuracy, above the majority-class floor, against 0.754 at 0.50.
F1 is flat across both. Accuracy improves here because the higher threshold
predicts fewer positives, and most clients are negatives.

**The threshold did not transfer cleanly from validation to test.** Validation
F1 at 0.61 was 0.533; the same threshold gives 0.515 on test. A threshold fitted
to one 6,000-row split is fitted partly to that split's noise.

**The precision floor held on validation and broke on test.** The 0.45 threshold
was chosen because it kept validation precision at 0.402, just above the 0.40
requirement. Test precision at that threshold is 0.396 — under the floor. A
constraint satisfied on a validation split is not a guarantee.

## Repo layout

dataset.py download, stratified 60/20/20 split, standardization, DataLoaders
model.py single-logit MLP, BCEWithLogitsLoss with pos_weight, optimizer
train.py training and validation loop
evaluate.py confusion matrix, precision/recall/F1, ROC AUC
threshold.py threshold sweep, operating-point comparison, checkpointing


Every module runs standalone and checks its own output:

```bash
python dataset.py
python model.py
python train.py
python evaluate.py
python threshold.py
```

The checkpoint stores the weights and the selected threshold together. Loading
the weights and re-applying 0.5 would discard the threshold selection entirely.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python threshold.py
```

Trains on CPU in a few minutes. The dataset downloads on the first run and is
cached locally.

## Data

**Default of Credit Card Clients** — 30,000 Taiwanese credit card clients from
April to September 2005. 23 features covering credit limit, demographics, six
months of repayment status, six months of bill amounts and six months of payment
amounts. Target is default in the following month. Fetched by `dataset.py` via
`ucimlrepo` (id=350) and not committed.

Splits are stratified, so train, validation and test all carry the same 22.1%
positive rate. The `StandardScaler` is fitted on the training split only;
validation and test are transformed with the training statistics.

Labels are `float32`, not `int64`, because `BCEWithLogitsLoss` compares against
float targets.

## Known limitations

**The threshold is selected on 6,000 validation rows containing 1,327
positives.** The F1 curve is flat near its maximum, so the selected value is not
tightly determined, and the validation-to-test drop of 0.018 F1 is consistent
with fitting split noise rather than a real optimum.

**Data is from one market in 2005.** Nothing here transfers to current lending
decisions elsewhere.

## Notes on authorship

A learning project, written from specifications as part of a PyTorch course in university.