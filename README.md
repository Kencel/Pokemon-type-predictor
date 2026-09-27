# Pokémon Type Predictor

> **Status:** Planned experiment. Settings below are the declared plan and will be updated (with a note) if anything changes before test evaluation.

CSCI 114 Lab 2: Testing whether reduced feature representations built with **PCA** and with an **autoencoder** can match or improve classification performance compared with training on the **full preprocessed feature set**. All models are implemented in **PyTorch**.

## Research question

Can a lower-dimensional representation of Pokémon attributes preserve (or even improve) primary-type classification performance relative to the full feature set? A well-supported finding that reduction does *not* help is also a valid outcome.

## Dataset

- **Source:** [Pokémon Stats Dataset (Kaggle, shreyashautomation)](https://www.kaggle.com/datasets/shreyashautomation/pokemon-stats-dataset), built from [PokéAPI](https://pokeapi.co). MIT license.
- **Size:** 1,351 rows × 20 columns: 1,025 species plus 326 alternate forms (Megas, regional variants, Gigantamax, etc.)
- **Task:** Multiclass classification of **primary type (`type_1`)**, 18 classes
- **Class distribution:** Imbalanced, from 160 (water) and 139 (normal) down to 35 (fairy) and 13 (flying)
- **Missing values:** `base_experience` is missing for 49 rows, all alternate forms

### Input features (n = 10, all numeric)

| Feature | Description |
|---|---|
| `hp`, `attack`, `defense`, `special_attack`, `special_defense`, `speed` | Base stats |
| `height_dm` | Height in decimetres |
| `weight_hg` | Weight in hectograms |
| `base_experience` | Experience gained for defeating the Pokémon |
| `capture_rate` | Base capture rate (higher = easier to catch) |

### Excluded columns

| Column(s) | Reason |
|---|---|
| `id`, `name` | Identifiers. `name` is used only to build the species grouping key for the split. |
| `type_1` | Target label |
| `type_2` | Type information; excluded to keep label-related information out of the inputs |
| `abilities` | Many abilities are type-specific (e.g., Blaze → fire, Torrent → water) and would effectively reveal the target |
| `generation`, `is_legendary`, `is_mythical`, `color`, `shape` | Categorical/boolean descriptors; excluded so PCA and the autoencoder compress the same kind of features |

### Why dimensionality reduction may help

The six base stats tend to rise together with overall strength, and `base_experience` and `capture_rate` also track stat power. `height_dm` and `weight_hg` are correlated and right-skewed. A few constructed features may capture most of the variation. The question is whether that compressed variation still separates types.

### Getting the data

The raw data is not included in this repository. Download it from the Kaggle page above and save the CSV as `data/pokemon_complete_stats.csv`.

## Experimental design

### 1. Split

- **Train / validation / test = 70 / 15 / 15**, made **before** any preprocessing is fitted.
- **Grouped by species:** alternate forms share stats and abilities with their base species, so every form of a species goes into the same partition. Forms (`id` ≥ 10000) are mapped to their base species by name prefix.
- **Stratified by `type_1`** as far as grouping allows.
- **Split seed:** 42 (fixed; every configuration uses the same observations).
- Rare classes (flying: 13 rows total) will have only a few validation and test examples; per-class results for them are noted as unstable.

### 2. Preprocessing (fit on the training set only)

1. Impute missing `base_experience` with the training-set median.
2. Apply `log1p` to `height_dm` and `weight_hg` to reduce right skew.
3. Standardize all 10 features (training-set mean and std).

The fitted transforms are then applied to the validation and test sets. Labels are never used as preprocessing, PCA, or autoencoder inputs.

### 3. Classifier (identical for every representation)

| Setting | Value |
|---|---|
| Architecture | input (width = n or k) → 64 → 64 → 18 |
| Hidden activation | ReLU |
| Output | 18 logits, softmax via `CrossEntropyLoss` |
| Optimizer | Adam, learning rate 1e-3 |
| Batch size | 64 |
| Max epochs | 300 |
| Stopping rule | Early stopping on validation cross-entropy (patience 20); restore best checkpoint |
| Seeds | 0, 1, 2, 3, 4 |

- **Primary metric: macro F1.** With 18 imbalanced classes, accuracy is dominated by common types such as water and normal. Macro F1 weights every type equally.
- **Supporting metrics:** accuracy, balanced accuracy, per-class F1, confusion matrix.
- A **fresh classifier** is trained for every representation and seed; only the input width changes.

### 4. Candidate dimensions

**k ∈ {2, 5, 8}** (small, intermediate, large; all < n = 10). The same values are used for PCA and the autoencoder.

**Selection rule (both methods):** choose the k with the highest mean validation macro F1. If a smaller k has a mean within one standard deviation of the best, prefer the smaller k.

### 5. PCA

- Implemented in PyTorch via SVD of the centered, standardized training inputs, and checked against `sklearn.decomposition.PCA`.
- Fit on training inputs only; validation and test inputs are projected with the fitted components.
- Report explained variance and component loadings for the selected components.

### 6. Autoencoder

| Setting | Value |
|---|---|
| Encoder | 10 → 32 → k (ReLU after the hidden layer, linear bottleneck) |
| Decoder | k → 32 → 10 (ReLU after the hidden layer, linear output) |
| Loss | MSE |
| Optimizer | Adam, learning rate 1e-3 |
| Batch size | 64 |
| Max epochs | 500 |
| Stopping rule | Early stopping on validation reconstruction MSE (patience 30); restore best checkpoint |
| Seeds | 0, 1, 2, 3, 4 (autoencoder retrained per seed) |

- **Linear output with MSE:** all inputs are standardized continuous values, so outputs are unbounded and MSE matches that scale.
- After training, the **encoder is frozen** and used to extract latent features for the classifier; the classification loss never updates the encoder.
- Reconstruction loss is used only for training and checkpoint selection, **not** for choosing k.

### 7. Test evaluation

After all choices are frozen, the baseline, the selected PCA configuration, and the selected autoencoder configuration are evaluated on the test set once per seed. All seeds are reported, and confusion matrices use **seed 0**, designated in advance.

### 8. Follow-up experiment (guide question 5, train/validation only)

Remove the redundant feature group `base_experience` and `capture_rate` (n = 10 → 8) and rerun the baseline, PCA, and autoencoder with everything else fixed. The hypothesis and predicted effect will be written down before running it.

## Analysis outputs

- Validation macro F1 vs. k for PCA and the autoencoder (mean ± std across seeds), with the baseline as a reference line
- PCA explained variance and loadings
- Autoencoder probes: correlations between latent features and inputs, plus a class-colored latent plot for k = 2
- Autoencoder reconstruction loss vs. k (reported only; not used to select k)
- Confusion matrices for the three final configurations

## Results

_To be filled in._

### Candidate dimensions (validation macro F1, mean ± std over 5 seeds)

| k | PCA | Autoencoder |
|---|---|---|
| 2 | – | – |
| 5 | – | – |
| 8 | – | – |

### Final comparison

| Configuration | Input dimensions | Validation macro F1 | Test macro F1 |
|---|---|---|---|
| All-feature baseline | 10 | – | – |
| PCA + classifier | – | – | – |
| Autoencoder + classifier | – | – | – |

## Repository structure (tentative)

```
.
├── data/            # Dataset CSV
├── notebooks/       # Exploration and experiments
├── src/             # Preprocessing, models, training code
├── results/         # Figures and metrics
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Package versions will be pinned in `requirements.txt` (torch, numpy, pandas, scikit-learn, matplotlib) once the environment is final.
