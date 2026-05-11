"""
data_loader.py
==============
Handles loading, cleaning, and preprocessing for both benchmark datasets:
  - Breast Cancer Wisconsin (binary classification)
  - Boston Housing (regression)

Returns PyTorch tensors ready for training, split into train / val / test sets.

Preprocessing strategy
-----------------------
Numeric continuous columns  →  z-score standardisation (zero mean, unit variance)
Binary / categorical columns →  kept as-is (already in a meaningful 0/1 or
                                 low-cardinality integer space; z-scoring would
                                 destroy the discrete semantics, e.g. CHAS in
                                 Boston Housing is a 0/1 river-boundary flag)

A column is treated as binary/categorical when its number of unique values is
≤ CATEGORICAL_UNIQUE_THRESHOLD (default 2).  Ordinal columns with small
cardinality (e.g. RAD in Boston Housing, values 1-24) fall above this threshold
and are z-scored like continuous features — which is standard practice.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List, Tuple

# Columns with this many or fewer unique values are treated as categorical
CATEGORICAL_UNIQUE_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_column_types(X: np.ndarray,
                          threshold: int = CATEGORICAL_UNIQUE_THRESHOLD
                          ) -> Tuple[List[int], List[int]]:
    """
    Classify each column as continuous or categorical/binary.

    Parameters
    ----------
    X         : raw feature matrix, shape (n_samples, n_features)
    threshold : columns with ≤ threshold unique values → categorical

    Returns
    -------
    continuous_cols  : list of column indices to z-score
    categorical_cols : list of column indices to leave unchanged
    """
    continuous_cols:  List[int] = []
    categorical_cols: List[int] = []
    for j in range(X.shape[1]):
        col = X[:, j]
        n_unique = len(np.unique(col[~np.isnan(col)]))
        if n_unique <= threshold:
            categorical_cols.append(j)
        else:
            continuous_cols.append(j)
    return continuous_cols, categorical_cols


def _selective_standardize(
        X_train: np.ndarray,
        X_val:   np.ndarray,
        X_test:  np.ndarray,
        continuous_cols: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score standardise only the continuous columns (fitted on X_train).
    Categorical / binary columns pass through unchanged.
    """
    X_tr  = X_train.copy()
    X_v   = X_val.copy()
    X_te  = X_test.copy()

    if continuous_cols:
        idx   = np.array(continuous_cols)
        mean  = X_tr[:, idx].mean(axis=0)
        std   = X_tr[:, idx].std(axis=0)
        std[std == 0] = 1.0          # guard against zero-variance columns

        X_tr[:, idx]  = (X_tr[:, idx]  - mean) / std
        X_v[:, idx]   = (X_v[:, idx]   - mean) / std
        X_te[:, idx]  = (X_te[:, idx]  - mean) / std

    return X_tr, X_v, X_te


def _split(X: np.ndarray, y: np.ndarray,
           val_frac: float = 0.15, test_frac: float = 0.15,
           random_state: int = 42):
    """Deterministic train / val / test split."""
    rng  = np.random.RandomState(random_state)
    idx  = rng.permutation(len(X))
    n    = len(X)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val:]

    return (X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx],
            X[test_idx],  y[test_idx])


def _to_tensors(*arrays):
    return [torch.tensor(a, dtype=torch.float32) for a in arrays]


# ---------------------------------------------------------------------------
# Breast Cancer Wisconsin
# ---------------------------------------------------------------------------

def load_breast_cancer(path: str) -> Dict[str, Any]:
    """
    Load the Breast Cancer Wisconsin dataset.

    Target: 'diagnosis'  →  M (malignant) = 1,  B (benign) = 0
    Drops the 'id' column and any unnamed trailing columns.

    Returns a dict with:
        X_train, y_train, X_val, y_val, X_test, y_test  – float32 tensors
        n_features   – int
        n_outputs    – int (1)
        task         – 'classification'
        feature_names – list[str]
    """
    df = pd.read_csv(path)

    # Drop non-feature columns
    drop_cols = [c for c in df.columns
                 if c.lower() == 'id' or 'unnamed' in c.lower()]
    df = df.drop(columns=drop_cols)

    feature_cols = [c for c in df.columns if c != 'diagnosis']
    X = df[feature_cols].values.astype(np.float32)
    y = (df['diagnosis'] == 'M').values.astype(np.float32)

    if np.isnan(X).any():
        # Median imputation (shouldn't be needed for this dataset)
        col_median = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_median, inds[1])

    # All Breast Cancer features are continuous measurements → standardise all
    continuous_cols, categorical_cols = _detect_column_types(X)

    X_train, y_train, X_val, y_val, X_test, y_test = _split(X, y)
    X_train, X_val, X_test = _selective_standardize(X_train, X_val, X_test,
                                                     continuous_cols)

    Xt, yt, Xv, yv, Xte, yte = _to_tensors(X_train, y_train, X_val, y_val,
                                             X_test,  y_test)
    return {
        'X_train':        Xt,
        'y_train':        yt.unsqueeze(1),
        'X_val':          Xv,
        'y_val':          yv.unsqueeze(1),
        'X_test':         Xte,
        'y_test':         yte.unsqueeze(1),
        'n_features':     X_train.shape[1],
        'n_outputs':      1,
        'task':           'classification',
        'feature_names':  feature_cols,
        'continuous_cols':  continuous_cols,
        'categorical_cols': categorical_cols,
    }


# ---------------------------------------------------------------------------
# Boston Housing
# ---------------------------------------------------------------------------

def load_boston_housing(path: str) -> Dict[str, Any]:
    """
    Load the Boston Housing dataset.

    Target: 'MEDV' (median house value).
    Handles missing values via per-column median imputation.

    Returns same structure as load_breast_cancer but with task='regression'.
    """
    df = pd.read_csv(path)

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    target_col   = 'MEDV'
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Median imputation for missing values
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = float(np.nanmedian(X[:, j]))

    y_missing = np.isnan(y)
    if y_missing.any():
        y[y_missing] = float(np.nanmedian(y[~y_missing]))

    # Detect binary/categorical columns BEFORE the split (uses full dataset for
    # unique-value counting, which is safe — we only look at cardinality, not
    # the values themselves, so there is no data leakage).
    continuous_cols, categorical_cols = _detect_column_types(X)

    X_train, y_train, X_val, y_val, X_test, y_test = _split(X, y)

    # Only z-score continuous columns; categorical (e.g. CHAS=binary 0/1) are
    # left as-is so their discrete meaning is preserved.
    X_train, X_val, X_test = _selective_standardize(X_train, X_val, X_test,
                                                     continuous_cols)

    # Also standardise y for regression to stabilise training
    y_mean = y_train.mean()
    y_std  = y_train.std() if y_train.std() > 0 else 1.0
    y_train_s = (y_train - y_mean) / y_std
    y_val_s   = (y_val   - y_mean) / y_std
    y_test_s  = (y_test  - y_mean) / y_std

    Xt, yt, Xv, yv, Xte, yte = _to_tensors(X_train, y_train_s,
                                             X_val,   y_val_s,
                                             X_test,  y_test_s)
    return {
        'X_train':        Xt,
        'y_train':        yt.unsqueeze(1),
        'X_val':          Xv,
        'y_val':          yv.unsqueeze(1),
        'X_test':         Xte,
        'y_test':         yte.unsqueeze(1),
        'n_features':     X_train.shape[1],
        'n_outputs':      1,
        'task':           'regression',
        'feature_names':  feature_cols,
        'continuous_cols':  continuous_cols,
        'categorical_cols': categorical_cols,
        'y_mean':         float(y_mean),
        'y_std':          float(y_std),
    }
