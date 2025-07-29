# FED3-data

## Project Purpose
This repository contains the full analysis pipeline for behavioural data collected with the **FED3** feeding device.  The goal is to compare feeding behaviour between control mice and mice with CASK knock-down (CASK-KD) across two learning paradigms:

* **FR1** – fixed-ratio schedule where every correct poke delivers one pellet.
* **Reversal** – identical schedule, but the active poke is periodically switched.

Raw data live in `data/` (Excel workbooks exported from Med-PC / FED3), while summary figures are written to `export/`.

All analyses are performed in Jupyter notebooks under `CASK_analysis/` (for CASK vs. Ctrl) and `WT_analysis/` (for single group test of pipeline).  The notebooks imported codebase is located in `scripts/`.

---

## Quick Start
1.  Clone the repo and install Python ≥3.9.
2.  Install requirements
   ```bash
   pip install -r requirements.txt
   ```
3.  Launch JupyterLab or VS Code in the project root and open any notebook in `CASK_analysis/` or `WT_analysis/`.

The notebooks can be run top-to-bottom; figures will appear in the notebook and be saved automatically inside `export/`.

---

## Use on Your Data Organization
For your own purpose or data format, please use your own usage interface but the analysis code generall compatible with Excel, csv, tsv formats and can adopt with various format with very small editions. If you want to directly use our interface, please organize the data in the same way as we did (two group comparison: `CASK_analysis`; one group value: `WT_analysis`). You can read our `data/` directory for data organization expectation.

For Python or FED3 data beginners, you can use VS Code with Python extension packs so that you can use your cursor to stop on each method imported frmo `scripts/` in the notebook to see the documentation of the method. 

---

## Notebook Guide (CASK cohort)
Below is a concise tour of each notebook and the core utilities it relies on.  For readability we refer to helper functionality conceptually instead of listing individual function names.

| Notebook | Main Task | Helper Functionality Used |
|----------|-----------|---------------------------|
| **Dispense check.ipynb** | Quality-control sanity check: quantifies how often the stepper motor rotated ≥15 times during a pellet event, a proxy for mechanical errors.  Outputs a JSON report per sheet. | • Centralised paths & sheet lists.<br>• Minimal in-notebook helper – no additional script dependency. |
| **Accurate Meal Model.ipynb** | Semi-supervised labelling of meals followed by training deep-learning classifiers that distinguish *accurate* vs *inaccurate* meals.  The workflow: <br> → extract raw meal sequences <br> → cluster them with K-means (elbow & silhouette diagnostics + manuall inspection) <br> → manually mark good / bad clusters <br> → train an LSTM and a 1-D CNN <br> → evaluate on held-out data and save the weights. | • Reading Excel sheets and extracting pellet sequences.<br>• Meal segmentation & padding utilities.<br>• End-to-end K-means helpers with PCA visualisation.<br>• Dataset builders that convert padded meals to tensors.<br>• Ready-made PyTorch models (LSTM, CNN) plus training / evaluation loops.<br>• Convenience I/O for saving pickles and `.npz` datasets. |
| **FR1 Analysis.ipynb** | Generates all FR1 figures/statistics (Figure 2).  Computes cumulative accuracy curves, first-learned line, pellet counts, first-meal latency, light- vs dark-phase meal ratios, and applies t-tests between groups. | • Data loader with on-the-fly cleaning (event renaming, accuracy calculation, retrieval-time parsing).<br>• High-resolution cumulative-accuracy plotter with SEM.<br>• Generic group-stats violin/box/strip plotter with automatic summary text.<br>• Meal utilities for pellet-frequency flips, first-meal detection, and active-phase classification.<br>• Statistical testing helper (independent / paired t-tests). |
| **Reversal Block.ipynb** | Comprehensive reversal-session analysis (Figure 3 & 4 plus Supplement).  Three sections: <br> (1) Block-wise transition plots and first-meal metrics, <br> (2) Learning-score computation across blocks, <br> (3) Pellet-retrieval latency and regression over block index. | • Block splitter that segments sessions whenever the active poke switches.<br>• Transition counter that quantifies Left/Right sequences, success rate, pellet rate, and inactive blocks.<br>• Meal detection inside each block including accuracy labelling.<br>• Learning-score and learning-result calculators that summarise behaviour across user-defined proportions of the block.<br>• Publication-quality plotting helpers for transition maps, learning-score trends, pellet-ratio trends, and group comparisons.<br>• Retrieval-time extractor, outlier removal, block-wise plotting, and linear-fit annotator. |
| **Reversal Feeding.ipynb** | Focuses on food-intake metrics during reversal (Figure 4 Supplement).  Re-computes pellet per hour, meal counts, first-meal latency, dark-phase meal proportion, and performs hypothesis testing. | • Same meal-processing pipeline as FR1.<br>• Group-stats plotting and t-test helper. |
| **Meal Pattern Distance.ipynb** | Exploratory comparison of meal patterns between groups.  Uses the previously trained LSTM to score every meal, then compares the score vectors with cosine similarity, Wasserstein distance, KL divergence, logistic regression, and K-means clustering.  Includes interactive 3-D visualisations. | • Fast extraction of meals ready for model input.<br>• Batch-mode model inference.<br>• Vectorisation helpers that summarise predictions by pellet count.<br>• Utilities for similarity metrics and dimensionality reduction visualisation. |

---

## Script Directory Overview (`/scripts`)
The notebooks import reusable helpers defined here.  Below is a high-level description of what each module provides. For detailed usage and documentations, please refer to `/scripts` folder and each method's docs.

* **preprocessing.py** – High-throughput Excel reader and cleaner.  Handles event renaming, time-stamp conversion, cumulative accuracy calculation, retrieval-time extraction, sheet discovery, and convenience helpers for mixed-type columns.
* **accuracy.py** – Tools for cumulative accuracy analytics, finding learning onset, plotting group statistics (violin/box/strip overlays), and cumulative pellet/meal counters.
* **meals.py** – Meal-level analytics: pellet-frequency histograms, cumulative-pellet plots, meal segmentation based on inter-pellet interval & count thresholds, first-meal detection, active-phase classification, and batch metrics suitable for model input.
* **meal_classifiers.py** – PyTorch pipelines: dataset wrapper, LSTM and 1-D CNN architectures tailored for variable-length meals, training loop with real-time accuracy, evaluation helpers, and parameter counters.
* **unsupervised_helpers.py** – Utilities for clustering unlabeled meals: K-means with elbow criterion and silhouette score, PCA visualisation, bookkeeping of good/bad meal indices, padding, dataset creation, and dataset merging.
* **direction_transition.py** – Reversal-specific analytics: splits sessions into blocks, counts event transitions, derives success rates, computes block-level pellet ratios, learning scores, learning results, and renders transition heatmaps and learning-score trends.
* **intervals.py** – Interval-focused helpers: cleans pellet-only CSV logs, computes mean/SEM pellet-retrieval times (with optional outlier removal), plots retrieval-time trend per block with best-fit regression line, and provides a flexible t-test wrapper.
* **organization.py** – Data-housekeeping utilities: batch convert CSV → Excel sheets, concatenate files, fix pellet counters when merging, and group sheet names into control/CASK cohorts.
* **path.py** – Central registry of file paths and ordered sheet lists (FR1 vs Reversal, male vs female, etc.) automatically populated from the Excel workbooks.

---

## Outputs
Running the notebooks will populate `export/` with publication-ready `.svg` figures grouped by paper figure number and supplementary panels.  Intermediate datasets (NumPy, pickle, PyTorch weights, JSON stats) are written to `data/` and `stats/`.

---

## Contributing
Feel free to open an issue or PR if you spot a bug, would like additional documentation, or wish to port the analysis to new datasets.

---

© 2024 FT LAB