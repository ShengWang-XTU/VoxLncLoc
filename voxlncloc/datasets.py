"""Loaders for the five official VoxLncLoc splits."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _clean(s: str) -> str:
    return str(s).upper().replace("U", "T").replace(" ", "")


@dataclass
class LoadedDataset:
    key: str
    task: str
    sequences: list[str]
    labels: np.ndarray
    label_names: list[str]
    split: dict


def _parse_lncdnn_fasta(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    seqs, labs, ids = [], [], []
    cur_id, cur_lab, chunks = None, None, []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_lab is not None and chunks:
                    seqs.append(_clean("".join(chunks)))
                    labs.append(cur_lab)
                    ids.append(cur_id or f"seq{len(ids)}")
                header = line[1:]
                if "_" in header and ("nucleolus" in header.lower() or "nucleoplasm" in header.lower()):
                    parts = header.rsplit("_", 1)
                    cur_id, lab_raw = parts[0], parts[-1]
                else:
                    toks = header.split()
                    cur_id = toks[0]
                    lab_raw = toks[-1] if len(toks) > 1 else toks[0]
                cur_lab = 1 if "nucleolus" in lab_raw.lower() else 0
                chunks = []
            else:
                chunks.append(line)
    if cur_lab is not None and chunks:
        seqs.append(_clean("".join(chunks)))
        labs.append(cur_lab)
        ids.append(cur_id or f"seq{len(ids)}")
    return seqs, np.asarray(labs, dtype=np.int64), ids


def load_lncdnn() -> LoadedDataset:
    d = DATA / "D_LncDNN"
    tr_s, tr_y, _ = _parse_lncdnn_fasta(d / "train.fasta")
    va_s, va_y, _ = _parse_lncdnn_fasta(d / "val.fasta")
    te_s, te_y, _ = _parse_lncdnn_fasta(d / "test.fasta")
    seqs = tr_s + va_s + te_s
    y = np.concatenate([tr_y, va_y, te_y])
    n_tr, n_va = len(tr_s), len(va_s)
    return LoadedDataset(
        key="D_LncDNN",
        task="binary",
        sequences=seqs,
        labels=y,
        label_names=["Nucleoplasm", "Nucleolus"],
        split={
            "train_idx": np.arange(n_tr),
            "val_idx": np.arange(n_tr, n_tr + n_va),
            "test_idx": np.arange(n_tr + n_va, n_tr + n_va + len(te_s)),
        },
    )


def load_gshape() -> LoadedDataset:
    d = DATA / "D_gShape"
    tr = pd.read_csv(d / "train.csv", sep="\t")
    te = pd.read_csv(d / "test.csv", sep="\t")
    df = pd.concat([tr, te], ignore_index=True)
    n_tr = len(tr)
    return LoadedDataset(
        key="D_gShape",
        task="binary",
        sequences=[_clean(x) for x in df["code"].tolist()],
        labels=df["Value"].astype(int).to_numpy(),
        label_names=["class0", "class1"],
        split={
            "train_idx": np.arange(n_tr),
            "test_idx": np.arange(n_tr, n_tr + len(te)),
            "protocol": "train_10fold_independent_test",
            "n_cv": 10,
            "cv_seed": 0,
        },
    )


def load_yi() -> LoadedDataset:
    df = pd.read_csv(DATA / "D_Yi" / "table6_miller_nomiddle_balanced.csv")
    y = df["label"].astype(np.int64).to_numpy()
    return LoadedDataset(
        key="D_Yi",
        task="binary",
        sequences=[_clean(x) for x in df["sequence"].tolist()],
        labels=y,
        label_names=["nucleus", "cytoplasm"],
        split={"protocol": "2x5fold", "cv_seed": 42, "n": int(len(y))},
    )


def _parse_fasta_labeled(path: Path) -> tuple[list[str], list[str]]:
    seqs, labs = [], []
    cur_lab, chunks = None, []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_lab is not None and chunks and cur_lab.lower() != "label":
                    seqs.append(_clean("".join(chunks)))
                    labs.append(cur_lab)
                parts = line[1:].split()
                cur_lab = parts[-1] if len(parts) >= 2 else parts[0]
                chunks = []
            else:
                chunks.append(line)
    if cur_lab is not None and chunks and cur_lab.lower() != "label":
        seqs.append(_clean("".join(chunks)))
        labs.append(cur_lab)
    return seqs, labs


def load_mgb() -> LoadedDataset:
    keep = {"Cytoplasm", "Nucleus", "Ribosome", "Cytosol"}
    d = DATA / "D_MGB"
    tr_s, tr_l = _parse_fasta_labeled(d / "Data_train.fasta")
    te_s, te_l = _parse_fasta_labeled(d / "Data_test.fasta")
    names = sorted(keep)
    name_to_i = {n: i for i, n in enumerate(names)}

    def _filter(seqs, labs):
        xs, ys = [], []
        for s, lab in zip(seqs, labs):
            if lab in name_to_i and len(s) >= 20:
                xs.append(s)
                ys.append(name_to_i[lab])
        return xs, np.asarray(ys, dtype=np.int64)

    tr_s, tr_y = _filter(tr_s, tr_l)
    te_s, te_y = _filter(te_s, te_l)
    seqs = tr_s + te_s
    y = np.concatenate([tr_y, te_y])
    return LoadedDataset(
        key="D_MGB",
        task="multiclass",
        sequences=seqs,
        labels=y,
        label_names=names,
        split={"train_idx": np.arange(len(tr_s)), "test_idx": np.arange(len(tr_s), len(seqs))},
    )


def load_grasp() -> LoadedDataset:
    d = DATA / "D_GRASP"
    meta = json.loads((d / "meta.json").read_text())
    df = pd.read_csv(d / "all_samples.csv")
    locations = list(meta["locations"])
    y = np.array(
        [[1 if lab in str(s).split(",") else 0 for lab in locations] for s in df["Label"]],
        dtype=np.int32,
    )
    folds = []
    for i in range(int(meta["k_folds"])):
        tr = np.load(d / f"fold{i}_train_idx.npy")
        va = np.load(d / f"fold{i}_valid_idx.npy")
        folds.append((tr, va))
    return LoadedDataset(
        key="D_GRASP",
        task="multilabel",
        sequences=[_clean(s) for s in df["Sequence"].tolist()],
        labels=y,
        label_names=locations,
        split={"folds": folds, "seed": int(meta.get("random_state", 41))},
    )


def load_dataset(name: str) -> LoadedDataset:
    key = name.lower().replace("-", "_")
    if key in ("d_lncdnn", "lncdnn", "lncdnn_nucleolus"):
        return load_lncdnn()
    if key in ("d_gshape", "gshape", "gshape_rnalight"):
        return load_gshape()
    if key in ("d_yi", "yi", "yi_lncatlas15"):
        return load_yi()
    if key in ("d_mgb", "mgb", "mgb_rnalocate2"):
        return load_mgb()
    if key in ("d_grasp", "grasp", "grasp_6k_8loc"):
        return load_grasp()
    raise KeyError(name)
