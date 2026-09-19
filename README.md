# VoxLncLoc

Codes and processed splits for VoxLncLoc, a multi-dataset lncRNA subcellular localization method based on tetrahedral 3D-CGR density voxels. Exact *k*-mer frequencies are provided as a compositional control.

## Requirements

```
pip install -r requirements.txt
```

Python 3.10 or later is recommended.

## Encoding parameters

Shared on all five official splits:

- contraction scales α = {0.3, 0.5, 0.7, 0.9}
- voxel grid *V* = 20
- Gaussian width σ = 0.45
- occupancy counts, then log(1 + *x*) and max-normalization
- arc-length and index resampling (both kept)
- PCA to 202 components, fitted on the training split only
- window length *W* = *n_pts*
- classifier seed 42; GRASP folds use seed 41

(*S*, *n_pts*) is selected per dataset:

| Dataset | Setting |
| --- | --- |
| D_LncDNN | *k* = 8, ExtraTrees |
| D_gShape | *S* = 3, *n_pts* = 512 concatenated with *k* = 5, random forest |
| D_Yi | *k* = 5, XGBoost (2 × 5-fold CV) |
| D_MGB | *S* = 3, *n_pts* = 512, LightGBM |
| D_GRASP | *S* = 1, *n_pts* = 256 concatenated with *k* = 8, ExtraTrees |

## Usage

The entry script is `main.py`. It trains the paper operating point on an official split and prints the corresponding metrics:

```
python main.py --dataset D_LncDNN
python main.py --dataset D_Yi
python main.py --dataset D_gShape
python main.py --dataset D_MGB
python main.py --dataset D_GRASP
```

D_LncDNN and D_Yi use the *k*-mer control only and finish relatively quickly. D_gShape, D_MGB, and D_GRASP encode 3D-CGR voxels and take longer.

The encoder can also be called from Python:

```python
from voxlncloc.datasets import load_dataset
from voxlncloc.encode import encode_voxel_matrix
from voxlncloc.kmers import exact_kmer_matrix
from voxlncloc.classifiers import make_classifier

ds = load_dataset("D_LncDNN")
K = exact_kmer_matrix(ds.sequences, k=8)
clf = make_classifier("et")
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
