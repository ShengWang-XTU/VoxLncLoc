#!/usr/bin/env python3
"""Run VoxLncLoc (3D-CGR voxels) on an official split.

Examples
--------
python main.py --dataset D_LncDNN
python main.py --dataset D_gShape
python main.py --dataset D_Yi
python main.py --dataset D_MGB
python main.py --dataset D_GRASP
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from voxlncloc.classifiers import make_classifier
from voxlncloc.datasets import load_dataset
from voxlncloc.encode import PCA_N, encode_voxel_matrix
from voxlncloc.evaluate import binary_metrics, multiclass_metrics, multilabel_metrics
from voxlncloc.kmers import exact_kmer_matrix

CONFIG = {
    "D_LncDNN": dict(n_seg=1, n_pts=256, head="et", k=8),
    "D_gShape": dict(n_seg=3, n_pts=512, head="xgb", k=5),
    "D_Yi": dict(n_seg=3, n_pts=512, head="xgb", k=5),
    "D_MGB": dict(n_seg=3, n_pts=512, head="lgbm", k=5),
    "D_GRASP": dict(n_seg=1, n_pts=256, head="et", k=8),
}

PRINT_KEYS = {
    "D_LncDNN": ["Precision", "Recall", "ACC", "F1-score", "AUC"],
    "D_gShape": ["AUROC", "ACC", "MCC", "F1", "Precision", "Recall"],
    "D_Yi": ["Accuracy", "F1", "MCC", "AUROC"],
    "D_MGB": ["Macro precision", "Macro recall", "Macro F1-score", "Macro accuracy", "AUC"],
    "D_GRASP": [
        "Example ACC",
        "Average Precision",
        "AvgF1",
        "Micro Precision",
        "Micro Recall",
        "AvgAUC",
        "AvgMCC",
        "Hamming Loss",
        "Ranking Loss",
        "One-Error",
        "Coverage",
    ],
}


def _proba(clf, X, task: str) -> np.ndarray:
    if task == "multilabel":
        raw = clf.predict_proba(X)
        cols = []
        for p in raw:
            p = np.asarray(p)
            cols.append(p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel())
        return np.column_stack(cols)
    p = np.asarray(clf.predict_proba(X))
    if task == "multiclass":
        return p
    if p.ndim == 2 and p.shape[1] > 1:
        return p[:, 1]
    return p.ravel()


def _make_clf(name: str, task: str, n_class: int, n_jobs: int):
    base = make_classifier(name, n_class=n_class, n_jobs=n_jobs)
    if task == "multilabel":
        return MultiOutputClassifier(base)
    return base


def _score(task: str, y, proba) -> dict:
    if task == "binary":
        return binary_metrics(y, proba)
    if task == "multiclass":
        return multiclass_metrics(y, proba)
    return multilabel_metrics(y, proba)


def _pack(Xv, Xk, tr, te):
    parts_tr, parts_te = [], []
    if Xv is not None:
        n_comp = min(PCA_N, len(tr), Xv.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        parts_tr.append(pca.fit_transform(Xv[tr]))
        parts_te.append(pca.transform(Xv[te]))
    if Xk is not None:
        parts_tr.append(Xk[tr])
        parts_te.append(Xk[te])
    return np.hstack(parts_tr), np.hstack(parts_te)


def _mean_std(rows: list[dict]) -> dict:
    out = {}
    keys = rows[0].keys()
    for k in keys:
        vals = np.asarray([r[k] for r in rows], dtype=float)
        out[k] = float(np.nanmean(vals))
        out[k + "_std"] = float(np.nanstd(vals, ddof=0))
    return out


def _print_row(name: str, metrics: dict, with_std: bool = False) -> None:
    keys = PRINT_KEYS[name]
    bits = []
    for k in keys:
        if with_std and k + "_std" in metrics:
            bits.append(f"{k}={metrics[k]:.4f}±{metrics[k + '_std']:.4f}")
        elif name in ("D_gShape", "D_Yi"):
            bits.append(f"{k}={metrics[k]:.4f}")
        else:
            bits.append(f"{k}={metrics[k]:.3f}")
    print("  " + "  ".join(bits), flush=True)


def run_dataset(name: str, *, n_jobs: int, verbose: bool, with_kmer: bool) -> dict:
    cfg = CONFIG[name]
    ds = load_dataset(name)
    print(
        f"[{name}] 3D-CGR voxels  S={cfg['n_seg']}  n_pts={cfg['n_pts']}  head={cfg['head']}",
        flush=True,
    )
    print(f"  n={len(ds.sequences)}  task={ds.task}", flush=True)
    print(f"  encoding voxels S={cfg['n_seg']} n_pts={cfg['n_pts']}", flush=True)
    Xv, _ = encode_voxel_matrix(
        ds.sequences,
        n_seg=cfg["n_seg"],
        n_pts=cfg["n_pts"],
        verbose=verbose,
    )
    Xk = exact_kmer_matrix(ds.sequences, cfg["k"]) if with_kmer else None

    n_class = int(np.max(ds.labels)) + 1 if ds.task == "multiclass" else 2
    split = ds.split or {}

    if ds.task == "multilabel" and split.get("folds"):
        rows = []
        for i, (tr, te) in enumerate(split["folds"]):
            Xtr, Xte = _pack(Xv, Xk, tr, te)
            clf = _make_clf(cfg["head"], ds.task, n_class, n_jobs)
            clf.fit(Xtr, ds.labels[tr])
            m = _score(ds.task, ds.labels[te], _proba(clf, Xte, ds.task))
            rows.append(m)
            print(f"  fold {i + 1}/{len(split['folds'])} AvgAUC={m['AvgAUC']:.3f}", flush=True)
        agg = _mean_std(rows)
        _print_row(name, agg, with_std=False)
        return agg

    if split.get("protocol") == "2x5fold" or name == "D_Yi":
        y = ds.labels
        rows = []
        idx = np.arange(len(y))
        for rep in range(2):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + rep)
            for tr, te in skf.split(idx, y):
                Xtr, Xte = _pack(Xv, Xk, tr, te)
                clf = _make_clf(cfg["head"], ds.task, n_class, n_jobs)
                clf.fit(Xtr, y[tr])
                rows.append(_score(ds.task, y[te], _proba(clf, Xte, ds.task)))
        agg = _mean_std(rows)
        _print_row(name, agg, with_std=True)
        return agg

    if split.get("protocol") == "train_10fold_independent_test":
        tr_all = np.asarray(split["train_idx"], dtype=int)
        te = np.asarray(split["test_idx"], dtype=int)
        rows = []
        kf = KFold(n_splits=int(split.get("n_cv", 10)), shuffle=True, random_state=int(split.get("cv_seed", 0)))
        for tr_rel, _va_rel in kf.split(tr_all):
            tr = tr_all[tr_rel]
            Xtr, Xte = _pack(Xv, Xk, tr, te)
            clf = _make_clf(cfg["head"], ds.task, n_class, n_jobs)
            clf.fit(Xtr, ds.labels[tr])
            rows.append(_score(ds.task, ds.labels[te], _proba(clf, Xte, ds.task)))
        agg = _mean_std(rows)
        _print_row(name, agg, with_std=False)
        return agg

    tr = np.asarray(split["train_idx"], dtype=int)
    te = np.asarray(split["test_idx"], dtype=int)
    if split.get("val_idx") is not None and len(np.asarray(split["val_idx"])):
        tr = np.concatenate([tr, np.asarray(split["val_idx"], dtype=int)])
    Xtr, Xte = _pack(Xv, Xk, tr, te)
    clf = _make_clf(cfg["head"], ds.task, n_class, n_jobs)
    clf.fit(Xtr, ds.labels[tr])
    m = _score(ds.task, ds.labels[te], _proba(clf, Xte, ds.task))
    _print_row(name, m, with_std=False)
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxLncLoc 3D-CGR voxel evaluation")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(CONFIG) + ["all"],
        help="official split, or all five",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true", help="less voxel-encoding progress")
    parser.add_argument(
        "--kmer",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    names = list(CONFIG) if args.dataset == "all" else [args.dataset]
    for name in names:
        run_dataset(name, n_jobs=args.n_jobs, verbose=not args.quiet, with_kmer=args.kmer)


if __name__ == "__main__":
    main()
