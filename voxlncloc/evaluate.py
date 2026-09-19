"""Official-split metrics used in the VoxLncLoc tables."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    coverage_error,
    f1_score,
    hamming_loss,
    label_ranking_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _binary_scores(y_true, proba) -> np.ndarray:
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim == 2:
        p = p[:, 1] if p.shape[1] > 1 else p.ravel()
    return p.ravel()


def binary_metrics(y_true, proba) -> dict:
    y = np.asarray(y_true).ravel().astype(int)
    p = _binary_scores(y, proba)
    pred = (p >= 0.5).astype(int)
    return {
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "ACC": float(accuracy_score(y, pred)),
        "F1-score": float(f1_score(y, pred, zero_division=0)),
        "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "MCC": float(matthews_corrcoef(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
        "AUROC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "Accuracy": float(accuracy_score(y, pred)),
        "F1": float(f1_score(y, pred, zero_division=0)),
    }


def multiclass_metrics(y_true, proba) -> dict:
    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(proba, dtype=np.float64)
    pred = p.argmax(axis=1)
    auc = float("nan")
    if p.ndim == 2 and p.shape[1] > 1 and len(np.unique(y)) > 1:
        try:
            auc = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
        except ValueError:
            auc = float("nan")
    return {
        "Macro precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "Macro recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "Macro F1-score": float(f1_score(y, pred, average="macro", zero_division=0)),
        "Macro accuracy": float(accuracy_score(y, pred)),
        "AUC": auc,
    }


def _one_error(y_true: np.ndarray, scores: np.ndarray) -> float:
    top = scores.argmax(axis=1)
    hit = y_true[np.arange(len(y_true)), top]
    return float(1.0 - np.mean(hit))


def _example_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    both = np.logical_and(y_true, y_pred).sum(axis=1).astype(np.float64)
    either = np.logical_or(y_true, y_pred).sum(axis=1).astype(np.float64)
    return float(np.mean(both / np.maximum(either, 1e-8)))


def _avg_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p_sum, r_sum, n_p, n_r = 0.0, 0.0, 0, 0
    for yt, yp in zip(y_true, y_pred):
        n_pos = int(yt.sum())
        if n_pos == 0:
            continue
        r_sum += float(yp[yt == 1].sum() / n_pos)
        n_r += 1
        n_hat = int(yp.sum())
        if n_hat > 0:
            p_sum += float(yt[yp == 1].sum() / n_hat)
            n_p += 1
    rec = r_sum / max(n_r, 1)
    prec = p_sum / max(n_p, 1)
    if rec + prec == 0:
        return 0.0
    return float(2 * rec * prec / (rec + prec))


def multilabel_metrics(y_true, proba, thres: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=np.int32)
    p = np.asarray(proba, dtype=np.float64)
    pred = (p >= thres).astype(int)
    aucs = []
    mccs = []
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j])) > 1:
            aucs.append(float(roc_auc_score(y[:, j], p[:, j])))
        mccs.append(float(matthews_corrcoef(y[:, j], pred[:, j])))
    return {
        "Example ACC": _example_accuracy(y, pred),
        "Average Precision": float(average_precision_score(y, p, average="weighted")),
        "AvgF1": _avg_f1(y, pred),
        "Micro Precision": float(precision_score(y, pred, average="micro", zero_division=0)),
        "Micro Recall": float(recall_score(y, pred, average="micro", zero_division=0)),
        "AvgAUC": float(np.mean(aucs)) if aucs else float("nan"),
        "AvgMCC": float(np.mean(mccs)) if mccs else float("nan"),
        "Hamming Loss": float(hamming_loss(y, pred)),
        "Ranking Loss": float(label_ranking_loss(y, p)),
        "One-Error": _one_error(y, p),
        "Coverage": float(coverage_error(y, p)),
    }
