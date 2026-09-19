#!/usr/bin/env python3
"""Encode two short sequences and print the voxel-vector shape."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voxlncloc.encode import encode_voxel_matrix


def main() -> None:
    seqs = [
        "ACGUACGUACGUACGUACGUACGUACGUACGU",
        "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT",
    ]
    X, _ = encode_voxel_matrix(seqs, n_seg=1, n_pts=256)
    print("S=1, n_pts=256 stacked voxels:", X.shape)


if __name__ == "__main__":
    main()
