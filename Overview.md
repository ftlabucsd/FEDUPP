# FEDUPP — Detailed Documentation

> Back to **[README](README.md)** for quick-start instructions.

---

## Table of Contents

- [Overview](#overview)
- [Behavioral Paradigms](#behavioral-paradigms)
- [Project Structure](#project-structure)
- [Pipeline Notebook Guide](#pipeline-notebook-guide)
  - [Setup (Steps 1–3)](#setup-steps-13)
  - [Part A: FR1 Analysis (Steps 4–8)](#part-a-fr1-analysis-steps-48)
  - [Part B: Reversal Learning (Steps 9–21)](#part-b-reversal-learning-steps-921)
  - [Part C: IPI Analysis (Steps 22–23)](#part-c-ipi-analysis-steps-2223)
  - [Part D: Data Export (Step 24)](#part-d-data-export-step-24)
- [Meal Quality Classification](#meal-quality-classification)
- [Script Modules Reference](#script-modules-reference)
  - [preprocessing.py](#preprocessingpy)
  - [accuracy.py](#accuracypy)
  - [meals.py](#mealspy)
  - [direction_transition.py](#direction_transitionpy)
  - [utils.py](#utilspy)
  - [meal_classifiers.py](#meal_classifierspy)
  - [unsupervised_helpers.py](#unsupervised_helperspy)
  - [advanced_analysis.py](#advanced_analysispy)
- [Output Files & Figures](#output-files--figures)
- [Customization Guide](#customization-guide)
- [Troubleshooting](#troubleshooting)

---

## Overview

FEDUPP is a Python toolkit for neuroscience researchers working with **FED3 (Feeding Experimentation Device 3)** data. It provides automated workflows to:

- Analyze operant conditioning tasks (FR1 and Reversal Learning)
- Quantify learning metrics (accuracy, learning curves, adaptation speed)
- Classify meal quality using machine learning (LSTM / CNN)
- Assess cognitive flexibility via block-transition analysis
- Generate publication-ready figures with statistical comparisons

### Key Capabilities

| Feature | Detail |
|---------|--------|
| Quality control | Detects hardware malfunctions; auto-filters bad sessions |
| FR1 analysis | Learning acquisition, accuracy milestones, feeding organization |
| Reversal learning | Block transitions, WSLS strategies, retrieval-time trends |
| Meal classification | Neural-network classifier (good vs. bad meals) |
| Statistics | Built-in t-tests, ANOVAs, and group comparisons |
| Reproducibility | Modular Python scripts + documented Jupyter notebook |

---

## Behavioral Paradigms

### FED3 Device Code

We open-sourced our C++ code for the FED3 device [here](./scripts/ClassicFED3_WithReversalTask.ino), covering both FR1 and Reversal sessions.

### FR1 (Fixed-Ratio 1)

Every correct nose poke delivers one pellet. Measures basic operant learning.

**Metrics**: cumulative accuracy, time to 80 % milestone, pellets/day, meal frequency and quality.

### Reversal Learning

After FR1, the correct poke side switches periodically, testing cognitive flexibility.

**Metrics**: block success rates, transition patterns (L→L, L→R, R→R, R→L), learning scores, retrieval speed, meal quality during contingency changes.

---

## Project Structure

```
FEDUPP/
├── pipeline.ipynb                  # Main analysis pipeline (start here)
├── Accurate Meal Model.ipynb       # Train custom meal classifiers
├── requirements.txt
├── group_map.json                  # Mouse → group assignments
│
├── sample_data/                    # Input: FED3 CSV files (one folder per mouse)
│
├── data/                           # Pre-trained ML models & labeled data
│   ├── CNN_from_CASK.pth
│   ├── LSTM_from_CASK.pth
│   └── *.pkl
│
├── figures/                        # Output: generated plots & exports
│   ├── <method>/FR1/
│   └── <method>/REV/
│
└── scripts/                        # Reusable Python modules
    ├── preprocessing.py
    ├── accuracy.py
    ├── meals.py
    ├── direction_transition.py
    ├── utils.py
    ├── meal_classifiers.py
    ├── unsupervised_helpers.py
    ├── advanced_analysis.py
    └── *.ino                       # FED3 device code
```

---

## Pipeline Notebook Guide

The notebook (`pipeline.ipynb`) contains **24 steps** across four parts. Each step is self-contained; run cells top-to-bottom.

### Setup (Steps 1–3)

| Step | What it does |
|------|-------------|
| **1** | Import libraries; check environment against `requirements.txt` |
| **2** | Configure paths, load sessions, build group catalog |
| **3** | Check dispenser motor performance; remove sessions with >20 % mechanical errors |

### Part A: FR1 Analysis (Steps 4–8)

| Step | What it does | Key outputs |
|------|-------------|-------------|
| **4** | Compute ending accuracy and learning time | `fr1_end_accuracy`, `fr1_learn_time`; cumulative accuracy plot |
| **5** | Violin plots + t-tests for accuracy and milestone | `fr1_overall_accuracy.svg`, `fr1_learning_milestone_time.svg` |
| **6** | Detect meals, classify quality (ML), compute per-session metrics | 5 meal metrics + per-session diagnostic plots |
| **7** | Violin plots + t-tests for all meal metrics | 5 meal figures |
| **8** | Stacked histogram of high- vs. low-accuracy meals over time | `fr1_meal_accuracy_distribution.svg` |

**FR1 metrics at a glance**: overall accuracy · 80 % milestone time · pellets/day · first meal latency · first good meal latency · in-meal pellet ratio · good meal proportion.

### Part B: Reversal Learning (Steps 9–21)

| Step | What it does | Key outputs |
|------|-------------|-------------|
| **9** | Pre-compute blocks and within-block meals (cached for all later steps) | `rev_session_analyses` |
| **10** | Per-block transition patterns, success rates, meal timing | Per-mouse transition plots |
| **11** | Group-level violin plots + t-tests for transition metrics; WSLS two-panel | 6+ violin figures |
| **12** | WSLS sliding-window analysis: within-group, first-vs-last, two-way ANOVAs | `rev_wsls_first_last_10.svg` |
| **13** | Learning Score (early 75 %) and Learning Result (last 25 %) | `rev_learning_scores`, `rev_learning_results` |
| **14** | Learning score trend, learning result distribution, pellet-in-meal ratio | 3 figures + t-tests |
| **15** | Per-mouse feature table → correlation heatmap | `rev_feature_correlation.svg` |
| **16** | Match vs. Mismatch FR1 active poke → reversal block accuracy | Block accuracy distribution plots |
| **17** | Export meal accuracy distribution raw data + histograms to CSV | CSV files |
| **18** | Compute retrieval-time summaries (mean, projection, slope) | Per-mouse retrieval plots |
| **19** | Violin plots + t-tests for retrieval metrics | 3 retrieval figures |
| **20** | Combined FR1 + Reversal meal meta-summary: histograms, tables, CSV | Summary tables + plots |
| **21** | Meal accuracy vs. dispense time scatter + regression | Correlation figure |

**Reversal metrics at a glance**: number of blocks · first good meal time · meal accuracy · learning score/result · Win-Stay / Lose-Shift · retrieval time (mean, slope, projection) · pellet-in-meal ratio · inactive meal fraction · average meal size.

### Part C: IPI Analysis (Steps 22–23)

| Step | What it does | Key outputs |
|------|-------------|-------------|
| **22** | Calculate inter-pellet intervals by position (2–12) for FR1 and Reversal | `ipi_data_fr1`, `ipi_data_rev` |
| **23** | One violin plot per group showing IPI at each pellet position | Per-group SVG figures |

### Part D: Data Export (Step 24)

| Step | What it does | Key outputs |
|------|-------------|-------------|
| **24** | Write every metric to a multi-sheet Excel file | `<method>_analysis_data_export.xlsx` |

---

## Meal Quality Classification

The `Accurate Meal Model.ipynb` notebook trains neural-network classifiers to distinguish high-quality from poor feeding bouts.

### Workflow

```
Extract meal sequences → K-means clustering → Manual cluster labeling
→ Train LSTM / CNN → Evaluate → Save weights
```

### Details

1. **Meal detection** — ≤60 s between pellets, ≥2 pellets per meal. Extracts between-pellet accuracy sequences (e.g., `[100, 100, 50, 100]` for a 5-pellet meal).
2. **Unsupervised clustering** — K-means on meals grouped by pellet count (3, 4, 5+). Elbow + Silhouette methods guide *K*.
3. **Manual annotation** — label clusters as "good" (consistently high or rising accuracy) or "bad" (inconsistent / low).
4. **Model training** — LSTM (2-layer, 400 hidden, ~99 % acc) and CNN (1-D conv + dropout, ~98 % acc). Binary: 0 = good, 1 = bad.
5. **Deployment** — pre-trained weights in `data/`. Used automatically by the pipeline; retrain on your own data via the notebook.

Training takes <30 s (LSTM) and <10 s (CNN) on Apple M1 CPU — no GPU required.

### Training Your Own Classifier

1. Open `Accurate Meal Model.ipynb`.
2. Load your reversal sessions (cells 1–3).
3. For each pellet count (3, 4, 5+): run elbow method → fit K-means → label clusters → save `.pkl`.
4. Train LSTM/CNN and evaluate.
5. Save weights: `torch.save(model.state_dict(), 'data/CNN_from_YOUR_NAME.pth')`.

---

## Script Modules Reference

All reusable functions live in `scripts/`. Click a module name to jump to its reference.

### `preprocessing.py`

Data loading and quality control.

| Item | Purpose |
|------|---------|
| `SessionKey` / `SessionData` | Metadata container and raw-data holder |
| `build_session_catalog(root, map)` | Scan data dir → organized session dict |
| `session_cache(root, map)` | Cached version (faster repeated runs) |
| `load_session_csv(path)` | Read FED3 CSV, clean columns, add accuracy |
| `motor_turn_summary(path, cutoff=15)` | Count dispenser errors, return proportion |
| `infer_session_type(df)` | Auto-detect FR1 vs. REV |
| `get_retrieval_time(path, day)` | Extract pellet retrieval durations |

```python
from scripts.preprocessing import build_session_catalog

SESSIONS, GROUPINGS = build_session_catalog("sample_data", "group_map.json")
```

### `accuracy.py`

Learning-curve analysis.

| Function | Purpose |
|----------|---------|
| `read_and_record(session, …)` | Process one session → final accuracy, milestone, binned df |
| `plot_cumulative_accuracy(dfs, labels, bin)` | Group learning curves with SEM bands |
| `find_learning_milestone(data, window, threshold)` | First time rolling accuracy ≥ threshold |

### `meals.py`

Feeding-pattern analysis and ML classification.

| Function | Purpose |
|----------|---------|
| `collect_meal_meta(…)` | High-level entry point for FR1 or REV meal analysis |
| `combine_meal_meta(metas, …)` | Merge FR1 + REV summaries |
| `find_meals_paper(data, …)` | Detect meal boundaries (paper or IPI method) |
| `find_meals_by_blocks(blocks, …)` | Block-aware meal detection (no cross-block meals) |
| `predict_meal_quality(meals, model)` | LSTM/CNN inference → good/bad |
| `calculate_low_accuracy_meal_ratio(…)` | Ratio of meals < accuracy cutoff |

For reversal sessions, always use block-based meal detection to guarantee consistency:

```python
from scripts.meals import find_meals_by_blocks

session_meals, meal_acc, block_meal_info = find_meals_by_blocks(blocks)
```

### `direction_transition.py`

Reversal learning and block analysis.

| Function | Purpose |
|----------|---------|
| `compute_session_analysis(data, …)` | **Main entry**: blocks + meals in one call |
| `split_data_to_blocks(data, day)` | Split session at active-poke switches |
| `get_transition_info(blocks, …)` | Per-block transitions, success rate, meal timing |
| `learning_score(blocks, …)` | Early-block adaptation accuracy |
| `learning_result(blocks, …)` | Late-block performance accuracy |
| `wsls_for_session_blocks(blocks)` | Session-level Win-Stay / Lose-Shift |
| `wsls_pellet_window_from_session_analyses(…)` | Sliding-window WSLS with first/last comparison |
| `block_retrieval_summary(blocks, …)` | Per-block retrieval time + linear trend |
| `plot_transition_stats(…)` | Per-mouse transition figure |
| `plot_learning_score_trend(…)` | Group learning-score curves |

```python
from scripts.direction_transition import compute_session_analysis

analysis = compute_session_analysis(
    data=session.raw.copy(), day_limit=3,
    meal_config=(60, 2), method="paper",
)
```

### `utils.py`

Statistics and visualization helpers.

| Function | Purpose |
|----------|---------|
| `perform_T_test(ctrl, exp, …)` | T-test → statistic, p-value, significance |
| `run_pairwise_tests(metric, name, pairs)` | Batch t-tests for group pairs |
| `plot_group_stats_wrapper(…)` | Violin plot + optional outlier removal |
| `collect_metric(name, mapping)` | Extract metric from nested dict |
| `calculate_interpellet_intervals_by_position(…)` | IPI by pellet position |
| `plot_interpellet_intervals_by_group_separate(…)` | Per-group IPI violin plots |

### `meal_classifiers.py`

Neural-network models for meal quality.

| Item | Purpose |
|------|---------|
| `RNNClassifier` | 2-layer LSTM |
| `CNNClassifier` | 1-D CNN with dropout |
| `train(model, …)` | Training loop (Adam optimizer) |
| `predict(model, input)` | Single inference (0 = good, 1 = bad) |

### `unsupervised_helpers.py`

Clustering and data prep for model training.

| Function | Purpose |
|----------|---------|
| `extract_meal_sequences(sessions, …)` | Accuracy sequences for all meals |
| `find_k_by_elbow(data)` | Elbow plot for optimal K |
| `fit_model_single(data, k, …)` | K-means → clustered meals |
| `data_padding(data)` | Pad sequences to fixed length |

### `advanced_analysis.py`

Specialized visualizations and correlations.

| Function | Purpose |
|----------|---------|
| `plot_fr1_meal_accuracy_distribution(…)` | High vs. low accuracy meals over FR1 session time |
| `plot_reversal_block_accuracy_distribution(…)` | Match vs. Mismatch FR1 influence on reversal blocks |
| `plot_meal_dispense_time_correlation(…)` | Meal accuracy vs. dispense delay scatter + regression |
| `export_meal_accuracy_distribution_data(…)` | Export raw + histogram data to CSV |

---

## Output Files & Figures

### FR1 (`figures/<method>/FR1/`)

| File | Description |
|------|-------------|
| `fr1_cumulative_accuracy.svg` | Learning curves with SEM |
| `fr1_overall_accuracy.svg` | Final accuracy distribution |
| `fr1_learning_milestone_time.svg` | Time to 80 % accuracy |
| `fr1_avg_pellets.svg` | Pellet consumption rate |
| `fr1_first_meal_time.svg` | First meal latency |
| `fr1_first_good_meal_time.svg` | First quality meal latency |
| `fr1_in_meal_ratio.svg` | Organized vs. scattered eating |
| `fr1_good_meal_ratio.svg` | Proportion of quality meals |
| `fr1_meal_accuracy_distribution.svg` | High vs. low accuracy meals over time |
| `meals/*.svg` | Per-session pellet frequency + cumulative plots |
| `interpellet_intervals/*.svg` | Per-group IPI distributions |

### Reversal (`figures/<method>/REV/`)

| File | Description |
|------|-------------|
| `rev_learning_score_overall.svg` | Adaptation curves |
| `rev_learning_score.svg` / `rev_learning_result.svg` | Score / result distributions |
| `rev_number_of_blocks.svg` | Block counts |
| `rev_first_good_meal_time.svg` | Adaptation speed per block |
| `rev_meal_accuracy.svg` | Meal quality during reversals |
| `rev_win_stay_lose_shift.svg` | WSLS two-panel |
| `rev_wsls_first_last_10.svg` | WSLS sliding-window comparison |
| `rev_retrieval_*.svg` | Retrieval time (mean, projection, slope) |
| `rev_feature_correlation.svg` | Cross-feature heatmap |
| `rev_block_acc_dist_*.svg` | Match vs. Mismatch accuracy |
| `transition/*.svg` | Per-mouse block figures |
| `retrieval/*.svg` | Per-mouse retrieval trends |
| `meals/*.svg` | Per-session meal diagnostics |
| `interpellet_intervals/*.svg` | Per-group IPI |

### Data Export (`figures/<method>/`)

| File | Description |
|------|-------------|
| `<method>_analysis_data_export.xlsx` | All metrics in one multi-sheet workbook |

All figures are SVG format with clear axis labels, group color coding, and statistical annotations.

---

## Customization Guide

### Analysis Parameters

Edit these in `pipeline.ipynb` Step 2 and throughout:

```python
MEAL_METHOD = 'paper'       # 'paper' or 'ipi'
REV_DAY_LIMIT = 3           # Days of reversal data to analyze
REV_MEAL_CONFIG = (60, 2)   # (time_threshold_sec, min_pellets)
remove_outlier_stds = 2.5   # Outlier trim for violin plots (-1 to disable)
```

### Add Groups

Update `group_map.json` and re-run — groups are auto-detected. Optionally customize `TEST_PAIRS`:

```python
TEST_PAIRS = [("control", "new_group"), ("experimental", "new_group")]
```

### Use a Custom Meal Classifier

Train in `Accurate Meal Model.ipynb`, then point `scripts/meals.py` to your weights:

```python
model.load_state_dict(torch.load("data/CNN_from_YOUR_NAME.pth"))
```

### Add Custom Metrics

Create a function in `scripts/`, import it in the notebook, and use `plot_group_stats_wrapper` to visualize:

```python
from scripts.custom_analysis import my_metric

results = {g: [my_metric(s) for s in sessions] for g, sessions in fr1_group_sessions.items()}
plot_group_stats_wrapper(results, "My Metric", "units", "my_metric.svg", fr1_figure_dir)
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| "No sessions found" | Wrong data directory or folder structure | Ensure CSVs are in `sample_data/<mouse_id>/` |
| Missing meal classifier | Model `.pth` file absent | Check `data/CNN_from_CASK.pth` exists, or train your own |
| Import errors | Missing packages | `pip install -r requirements.txt` |
| High memory usage | Cached data | Call `session_cache.cache_clear()` in Step 2 |
| Empty reversal results | No REV sessions | Ensure mice have reversal CSV files |
| Step 3 shows flagged sessions | Hardware malfunctions | Sessions are auto-removed if >20 % errors |

### CSV Format Requirements

Required columns: `MM:DD:YYYY hh:mm:ss`, `Event`, `Active_Poke`, `Left_Poke_Count`, `Right_Poke_Count`.

- **Event** values: `"Left"`, `"Right"`, `"Pellet"`, or compound types like `"LeftWithDispense"`.
- **Active_Poke** values: `"Left"` or `"Right"`.

### Group Map Errors

If you see `KeyError: mouse_id`:
1. Confirm `group_map.json` lists **all** mice in `sample_data/`.
2. Folder names must match exactly (case-sensitive).
3. Validate JSON at [jsonlint.com](https://jsonlint.com/).

---

*Back to **[README](README.md)** · [Top of page](#fedupp--detailed-documentation)*
