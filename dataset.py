import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ucimlrepo import fetch_ucirepo

import os

CACHE = "credit_default_cache.parquet"

def load_raw():
    if os.path.exists(CACHE):
        df = pd.read_parquet(CACHE)

    else:
        ds = fetch_ucirepo(id=350)
        df = ds.data.features.copy()
        df["target"] = ds.data.targets.iloc[:,0].to_numpy()
        df.to_parquet(CACHE)
    
    X_df = df.drop(columns=["target"])
    y = df["target"]
    return X_df, y


def prepare_data():
    X_df, y = load_raw()

    # STAGE 2 - INSPECT THE IMBALANCE

    class_counts = y.value_counts()
    frac_default = (y == 1).mean()

    # STAGE 3 - STRATIFIED 3-WAY SPLIT (60 / 20 / 20)

    X_temp, X_test, y_temp, y_test = train_test_split(X_df,y,test_size=0.20, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp,y_temp, test_size=0.25, random_state=42, stratify=y_temp)


    # STAGE 4 - STANDARDIZE (FIT ON TRAIN ONLY)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # STAGE 5 - TENSORS, TensorDataset, DATALOADERS


    X_train_tensor = torch.tensor(X_train_s, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_s, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_s, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    val_ds = TensorDataset(X_val_tensor, y_val_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, batch_size=64)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, batch_size=64)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, batch_size=64)

    return {"train_loader": train_loader,
        "val_loader":   val_loader,
        "test_loader":  test_loader,
        "n_features":   X_df.shape[1],
        "frac_default": frac_default,
        "scaler":       scaler,        
        "y_test":       y_test}

# ------------------------------------------------------------------------------
# SYSTEM VERIFICATION BLOCK
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        data = prepare_data()
        xb, yb = next(iter(data["train_loader"]))
        assert xb.dtype == torch.float32 and yb.dtype == torch.float32
        assert xb.shape[1] == data["n_features"]
        print("Step 1 passed.")
    except AssertionError as e:
        print("Step 1 not yet correct:", e)
    except Exception as e:
        print("Something errored:", repr(e))
