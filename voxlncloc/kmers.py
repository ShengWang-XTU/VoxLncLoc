"""Exact k-mer frequencies used only as a compositional control."""
from __future__ import annotations

from typing import Sequence

import numpy as np

_ALPH = "ACGT"
_MAX_LEN = 8000


def _vocab(k: int) -> list[str]:
    if k == 1:
        return list(_ALPH)
    prev = _vocab(k - 1)
    return [p + a for p in prev for a in _ALPH]


def exact_kmer_matrix(sequences: Sequence[str], k: int, *, max_len: int = _MAX_LEN) -> np.ndarray:
    vocab = _vocab(int(k))
    index = {mer: i for i, mer in enumerate(vocab)}
    X = np.zeros((len(sequences), len(vocab)), dtype=np.float32)
    for r, seq in enumerate(sequences):
        s = str(seq).upper().replace("U", "T")[:max_len]
        if len(s) < k:
            continue
        row = X[r]
        for i in range(len(s) - k + 1):
            mer = s[i : i + k]
            j = index.get(mer)
            if j is not None:
                row[j] += 1.0
        tot = float(row.sum())
        if tot > 0:
            row /= tot
    return X
