# VoxLncLoc

Prediction of lncRNA subcellular localization based on 3D chaos game representation voxels.

Codes and processed splits for **VoxLncLoc**. The method maps lncRNA sequences to tetrahedral **3D-CGR occupancy voxels**.

## Requirements

```
pip install -r requirements.txt
```

Python 3.10 or later is recommended.

## 3D-CGR voxel encoding

Shared on all five official splits:

- tetrahedron vertices for A/C/G/T
- contraction scales α = {0.3, 0.5, 0.7, 0.9}
- voxel grid *V* = 20
- Gaussian width σ = 0.45
- occupancy counts, then log(1 + *x*) and max-normalization
- arc-length and index resampling (both kept)
- PCA to 202 components, fitted on the training split only
- window length *W* = *n_pts*
- classifier seed 42; GRASP folds use seed 41

Voxel grain (*S*, *n_pts*) and the tabular head used by `main.py`:

| Dataset | *S* | *n_pts* | Head |
| --- | --- | --- | --- |
| D_LncDNN | 1 | 256 | ExtraTrees |
| D_gShape | 3 | 512 | XGBoost |
| D_Yi | 3 | 512 | XGBoost |
| D_MGB | 3 | 512 | LightGBM |
| D_GRASP | 1 | 256 | ExtraTrees |

## Usage

`main.py` encodes 3D-CGR voxels and evaluates them on an official split:

```
python main.py --dataset D_LncDNN
python main.py --dataset D_gShape
python main.py --dataset D_Yi
python main.py --dataset D_MGB
python main.py --dataset D_GRASP
```

From Python:

```python
from voxlncloc.datasets import load_dataset
from voxlncloc.encode import encode_voxel_matrix
from voxlncloc.classifiers import make_classifier

ds = load_dataset("D_LncDNN")
X, pca = encode_voxel_matrix(ds.sequences, n_seg=1, n_pts=256, fit_pca=True)
clf = make_classifier("et")
clf.fit(X[ds.split["train_idx"]], ds.labels[ds.split["train_idx"]])
```

## Data

Processed official splits are in `data/` (see `data/README.md`). Please cite the original dataset papers:

- D_LncDNN: Li et al., Mol Ther Nucleic Acids, 2025
- D_gShape: Yu et al., IEEE Trans Comput Biol Bioinform, 2025
- D_Yi: Yi et al., Noncoding RNA, 2025
- D_MGB: Hu et al., BMC Biol, 2025
- D_GRASP: Hao et al., Brief Bioinform, 2026

Ablation tables are in `results/ablation_by_dataset/`.

## License

MIT for the source code in this repository. Sequence data remain under the terms of their original sources.
