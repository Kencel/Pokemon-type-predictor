# Pokémon Type Predictor

> **Status:** Tentative. This README describes the planned experiment and will be updated as the work progresses.

CSCI 114 Lab 2: Testing whether a reduced feature representation, built with **PCA** and/or an **autoencoder**, can match or improve classification accuracy compared with training on the **original full-feature dataset**. All models are implemented in **PyTorch**.

## Research question

Can a lower-dimensional representation of Pokémon stats preserve (or even improve) classification performance relative to the full feature set?

## Dataset

- **Source:** [Pokémon Stats Dataset (Kaggle, shreyashautomation)](https://www.kaggle.com/datasets/shreyashautomation/pokemon-stats-dataset)
- **Size:** ~1,351 Pokémon across multiple generations
- **Contents:** Base stats, abilities, and typings
- **Target (tentative):** Primary type (`Type 1`). Other candidate labels (e.g. legendary status) may be considered.

The raw data is not included in this repository. Download it from Kaggle and place the CSV in `data/`.

## Planned approach

1. **Preprocessing**
   - Drop identifiers and non-predictive columns (name, Pokédex number, etc.)
   - Encode categorical features, standardize numeric features
   - Stratified train / validation / test split with a fixed random seed
2. **Baseline:** Classifier (MLP) trained on the full feature set
3. **PCA:** Fit PCA on the training split only, then train the same classifier on the top *k* components for several values of *k*
4. **Autoencoder:** Train an autoencoder on the training split, then train the same classifier on the bottleneck (latent) features for several latent sizes
5. **(Optional) Combined:** PCA followed by an autoencoder, or other variants

To keep the comparison fair, the classifier architecture, training budget, and splits stay the same across all representations. Only the input features change.

## Evaluation

- Test accuracy (primary metric)
- Macro F1 score (to account for class imbalance across types)
- Mean and standard deviation over multiple random seeds
- PCA explained variance vs. number of components
- Autoencoder reconstruction loss vs. latent dimension

## Results

_To be filled in._

| Representation | Dimensions | Accuracy | Macro F1 |
|---|---|---|---|
| Full features | – | – | – |
| PCA | – | – | – |
| Autoencoder | – | – | – |

## Repository structure (tentative)

```
.
├── data/            # Dataset CSV (not tracked)
├── notebooks/       # Exploration and experiments
├── src/             # Preprocessing, models, training code
├── results/         # Figures and metrics
└── README.md
```

## Setup (tentative)

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas scikit-learn matplotlib
```
