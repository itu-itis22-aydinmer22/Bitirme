# Integrating Machine Learning and Signal Processing for Arrhythmia Detection

ITU graduation project pipeline implementing the system described in the
interim report (Şahin / Aydın / Demir, advisor Prof. Dr. Nizamettin Aydın,
January 2026).

## What this code does

End-to-end ECG arrhythmia detector:

1. **Signal pre-processing** (`src/signal_processing.py`) — 0.5–40 Hz Butterworth
   band-pass + Daubechies `db4` discrete wavelet denoising (universal
   threshold via MAD).
2. **R-peak detection** (`src/r_peak_detection.py`) — Pan-Tompkins 1985:
   bandpass → derivative → squaring → moving-window integration → adaptive
   thresholding, with R-peak refinement.
3. **Feature extraction** (`src/feature_extraction.py`) — hybrid feature
   vector around each R-peak: temporal (pre-RR / post-RR / QRS / PQ / QT /
   ST intervals), amplitude (P, Q, R, S, T peaks), morphological (5 PCA
   components on the QRS window).
4. **AAMI mapping + data loader** (`src/data_loader.py`) — collapses the
   raw MIT-BIH annotations into the 5 AAMI super-classes (N, S, V, F, Q).
5. **Training** (`src/train.py`) — Random Forest (200 trees, balanced
   class-weights) with **SMOTE** on each training fold and
   **GroupKFold** so the same patient never appears in both train and test.
6. **Evaluation** (`src/evaluate.py`) — aggregated confusion matrix,
   classification report, per-fold metric plots, feature-importance plot.
7. **Robustness** (`src/robustness.py`) — additive Gaussian noise at
   20 / 15 / 10 / 5 / 0 dB SNR, scaled per-feature by the training-set
   standard deviation so the dB level is comparable to a time-domain SNR.

## Running

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # numpy, scipy, scikit-learn, pywavelets,
                                         # pandas, matplotlib, seaborn,
                                         # imbalanced-learn, joblib, lightgbm,
                                         # xgboost, reportlab, pypdf
# (LightGBM/XGBoost on macOS additionally need: brew install libomp)

python main.py --n-estimators 200 --n-splits 5
```

### Publication-grade extensions (v2)

The pipeline was extended with a publication-quality benchmark suite. Each
experiment script lives under `experiments/` and writes its results to
`results/`:

| Script | What it does |
| --- | --- |
| `experiments/dechazal_ds1_ds2.py` | Single deterministic DS1-train / DS2-test split per de Chazal 2004 — runs LDA / RF (SMOTE & Borderline-SMOTE) / LightGBM / XGBoost / RF+HMM. |
| `experiments/benchmark_models.py` | 13-config grid: every (model, resampling, weighting, drop-Q, HMM) combination on 3-fold GroupKFold. |
| `experiments/cross_dataset.py` | Train on MIT-BIH, evaluate on the Supraventricular and INCART databases — quantifies cross-dataset generalisation. |
| `experiments/ablation.py` | Knock out one design decision at a time (feature group, single lead, no SMOTE, drop-Q, HMM, extended HRV/cross-lead features). |
| `experiments/multi_seed.py` | Same configuration, three random seeds, mean ± std of every metric. |
| `experiments/stacked_ensemble.py` | RF + LightGBM + XGBoost out-of-fold probabilities → logistic meta-learner with patient-disjoint stacking. |

All extension code is under `src/`:

| Module | Purpose |
| --- | --- |
| `src/models.py` | Unified factory for RF / LightGBM / XGBoost / LDA / LogReg with cost-sensitive sample weights (clinical AAMI cost matrix) and label-safe XGBoost wrapper. |
| `src/resampling.py` | Uniform interface for SMOTE / Borderline-SMOTE / ADASYN / SMOTE+Tomek / SMOTE+ENN / random-undersampling. |
| `src/postprocessing.py` | Patient-aware Hidden Markov Model Viterbi smoothing of beat-level predictions. |
| `src/extended_features.py` | Local-window HRV (SDNN / RMSSD / pNN50 approximations), cross-lead deltas, RR-interval ratios — engineered from the published 32-feature CSV without needing the raw waveform. |
| `src/ensemble.py` | Honest patient-disjoint stacked ensemble (RF + LGBM + XGB → LogReg meta). |
| `src/error_analysis.py` | Per-patient performance breakdowns + misclassification visualisations. |
| `src/data_loader.py` | Now also exposes `DS1_RECORDS` / `DS2_RECORDS` (de Chazal split), `feature_group_mask()` for ablation, `drop_q` flag. |
| `src/train.py` | Multi-model CV driver (any model × any resampling × optional HMM × `groupkfold`/`dechazal` fold strategy). |

Outputs land under `results/`:

| File | What it is |
| --- | --- |
| `class_distribution.csv` | AAMI 5-class beat counts |
| `fold_metrics.csv` | per-fold accuracy / macro-F1 / per-class F1 / latency |
| `fold_metrics_full.json` | same plus full confusion matrix per fold |
| `classification_report.txt` | sklearn report on aggregated CV predictions |
| `feature_importance.csv` | mean RF feature importance across folds |
| `robustness.json` | accuracy + macro-F1 + V-class F1 at each SNR |
| `figures/*.png` | confusion matrix, fold metrics, feature importance, class distribution |
| `models/rf_model.joblib`, `models/scaler.joblib` | final model trained on the full dataset |

## Data

The `*.csv` files are **not committed** (too large for GitHub). They are the
pre-extracted feature versions of the PhysioNet databases — columns match the
report's hybrid feature design exactly: `pre-RR`, `post-RR`, `pPeak`, `tPeak`,
`rPeak`, `sPeak`, `qPeak`, `qrs_interval`, `pq_interval`, `qt_interval`,
`st_interval`, `qrs_morph[0..4]` for each of the two leads (32 features total).

Place these files at the project root before running `main.py`:

| File | Size | Source |
| --- | --- | --- |
| `MIT-BIH Arrhythmia Database.csv` | ~45 MB | [Kaggle: shayanfazeli/heartbeat](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) (or PhysioNet WFDB + `src/feature_extraction.py`) |
| `MIT-BIH Supraventricular Arrhythmia Database.csv` | ~80 MB | same |
| `INCART 2-lead Arrhythmia Database.csv` | ~50 MB | same |
| `Sudden Cardiac Death Holter Database.csv` | ~180 MB | same |

The signal-processing and feature-extraction modules under `src/` reproduce
the same feature definitions on raw WFDB recordings so the same trained
classifier can be deployed on a fresh Holter stream.

## Work-package mapping (from §5.2 of the interim report)

| WP | Lead | Code |
| --- | --- | --- |
| WP1 — Signal Processing | Mehmet Eren Şahin | `signal_processing.py`, `r_peak_detection.py` |
| WP2 — Feature Extraction | Melih Demir | `feature_extraction.py`, `data_loader.py` |
| WP3 — Machine Learning | Mert Aydın | `train.py`, `evaluate.py`, `robustness.py`, `main.py` |
