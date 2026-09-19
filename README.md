# VoxLncLoc

Codes and processed official splits for **VoxLncLoc**, a multi-dataset lncRNA subcellular-localization pipeline based on tetrahedral 3D-CGR density voxels. Exact *k*-mer frequencies are included only as a compositional control.

Manuscript data-availability URL: https://github.com/ShengWang-XTU/VoxLncLoc

## Encoding constants (shared on all five splits)

| Parameter | Setting |
| --- | --- |
| Contraction scales α | {0.3, 0.5, 0.7, 0.9} |
| Voxel grid V | 20 |
| Gaussian width σ | 0.45 |
| Occupancy | integer counts, then log(1 + x) and max-normalization |
| Resampling | arc-length and index, both kept |
| PCA | 202 components, fitted on the training split only |
| Window vs. point budget | W = n_pts |
| Classifier seed | 42 |
| GRASP fold seed | 41 |

The voxel grain (S, n_pts) is chosen per dataset from training-side ablations. Operating points used in the comparison tables:

- D_LncDNN: *k* = 8 / ExtraTrees
- D_gShape: S = 3, n_pts = 512 voxels ⊕ *k* = 5 / random forest
- D_Yi: *k* = 5 / XGBoost (2 × 5-fold CV)
- D_MGB: S = 3, n_pts = 512 / LightGBM
- D_GRASP: S = 1, n_pts = 256 voxels ⊕ *k* = 8 / ExtraTrees

## Layout

```
voxlncloc/          voxel encoder, exact k-mers, tabular heads, split loaders
data/               processed official splits (see data/README.md)
results/            ablation CSVs behind Supplementary Tables 1–15
paper/              Supplementary Tables 1–15
scripts/            small usage example
```

Large voxel caches from internal sweeps are **not** uploaded (they exceed GitHub file limits). Re-encode from the FASTA/CSV files in `data/`.

## Install

```bash
python -m pip install -r requirements.txt
```

Smoke-test the encoder:

```bash
python scripts/example_encode.py
```

Load an official split:

```python
from voxlncloc.datasets import load_dataset
from voxlncloc.encode import encode_voxel_matrix
from voxlncloc.kmers import exact_kmer_matrix
from voxlncloc.classifiers import make_classifier

ds = load_dataset("D_LncDNN")
X, pca = encode_voxel_matrix(ds.sequences, n_seg=1, n_pts=256, fit_pca=True)
K = exact_kmer_matrix(ds.sequences, k=8)
clf = make_classifier("et")
```

## Dataset sources

Processed copies in `data/` are redistributed for reproducibility of the paper splits. Please cite the original releases:

- D_LncDNN — Li et al., *Mol Ther Nucleic Acids* 2025.
- D_gShape — Yu et al., *IEEE TCBB* 2025 (RNAlight train/test CSVs).
- D_Yi — Yi et al., *Noncoding RNA* 2025 (Miller unfiltered balanced Table 6 CSV).
- D_MGB — Hu et al., *BMC Biol* 2025 (Zenodo train/test FASTA).
- D_GRASP — Hao et al., *Brief Bioinform* 2026 (6k/8-loc multi-label set; folds with seed 41).

## License

MIT (this repository’s scripts). Underlying sequence databases remain under their original terms.
