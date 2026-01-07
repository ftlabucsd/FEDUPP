# FEDUPP Introduction

---

## 🎯 Overview

FEDUPP is a complete Python-based analysis toolkit designed for neuroscience researchers working with **FED3 (Feeding Experimentation Device 3)** data. This package provides automated workflows to:

- **Analyze operant conditioning tasks** (FR1 and Reversal Learning paradigms)
- **Quantify learning metrics** (accuracy, learning curves, adaptation speed)
- **Classify meal quality** using machine learning (LSTM/CNN models)
- **Assess cognitive flexibility** via block-transition analysis
- **Generate publication-ready figures** with statistical comparisons

### Key Features

✅ **Automated Data Quality Control** - Detects hardware malfunctions and filters problematic sessions  
✅ **Comprehensive FR1 Analysis** - Learning acquisition, accuracy milestones, feeding organization  
✅ **Reversal Learning Suite** - Block transitions, adaptation metrics, retrieval time trends  
✅ **Neural Network Meal Classifier** - Distinguishes high-quality vs poor feeding bouts  
✅ **Statistical Testing** - Built-in t-tests and group comparisons  
✅ **Reproducible & Modular** - Well-documented Jupyter notebooks + reusable Python modules  

---

## 📊 Behavioral Paradigms

### FED3 Code
We open-sourced our C++ code for programming FED3 device [here](./scripts/ClassicFED3_WithReversalTask_copy_20251001154805.ino) in FR1 and Reversal Sessions. 

### FR1 (Fixed-Ratio 1)
Every correct nose poke immediately delivers one pellet. Measures basic operant learning acquisition.

**Key Metrics Analyzed:**
- Cumulative accuracy over time
- Time to reach 80% accuracy milestone
- Pellets per hour
- Meal frequency and quality

### Reversal Learning
After FR1 training, the "correct" active poke side periodically switches, testing cognitive flexibility.

**Key Metrics Analyzed:**
- Block-by-block success rates
- Transition patterns (L→L, L→R, R→R, R→L)
- Learning scores (early block adaptation)
- Pellet retrieval speed trends
- Meal quality (inaccurate & accurate) during behavioral contingencies (blocks)

---

## 📁 Project Structure

```
FED3-data/
│
├── pipeline.ipynb      # ⭐ Main analysis pipeline (start here!)
├── Accurate Meal Model.ipynb # Train custom meal classifiers
├── requirements.txt           # Python dependencies
├── group_map.json             # Group assignments
│
├── sample_data/               # Input: Your FED3 CSV files
│   ├── M1/, M2/, M3/, ...
│
├── data/                      # Pre-trained ML models and sample labeled data
│   ├── CNN_from_CASK.pth
│   ├── LSTM_from_CASK.pth
│   └── [labeled meal data .pkl files]
│
├── figures/                   # Output: Generated plots
│   ├── FR1/
│   │   ├── cumulative_accuracy.svg
│   │   ├── overall_accuracy.svg
│   │   ├── meals/
│   │   └── ...
│   │   
│   └── REV/
│       ├── rev_learning_score_overall.svg
│       ├── transition/
│       ├── retrieval/
│       ├── meals/
│       └── ...
│       
└── scripts/                   # backbone scripts (imported by notebooks)
    ├── preprocessing.py       # Data loading & quality control
    ├── accuracy.py            # Learning curve analysis
    ├── meals.py               # Feeding pattern analysis & ML classification
    ├── direction_transition.py # Reversal learning & block analysis
    ├── utils.py               # Statistics & visualization helpers
    ├── meal_classifiers.py    # Neural network models (LSTM/CNN)
    ├── unsupervised_helpers.py # Clustering & data prep for model training
    ├── advanced_analysis.py   # Specialized analyses (FR1 influence, dispense timing)
    └── ClassicFED3_WithReversalTask_copy_20251001154805.ino  # FED3 device programming code
```

---

## 📓 Pipeline Notebook Guide

The `pipeline.ipynb` is organized into **19+ sequential steps** across four main sections:

### 🔧 Setup & Quality Control (Steps 1-3)

| Step | Description | Output |
|------|-------------|--------|
| **1** | Import libraries and helper functions | Ready environment |
| **2** | Load session catalog and group assignments | `SESSIONS`, `GROUPINGS` dictionaries |
| **3** | Check dispenser motor performance | Remove sessions with >20% mechanical errors |

### 📈 Part A: FR1 Analysis (Steps 4-7.5)

| Step | Description | Key Outputs |
|------|-------------|-------------|
| **4** | Compute learning metrics | `fr1_overall_accuracy`, `fr1_learning_milestone` |
| **5** | Visualize FR1 performance | Accuracy & milestone plots + t-tests |
| **6** | Analyze meal patterns | Pellet rates, meal timing, quality metrics |
| **6.5** | Print low-accuracy meal statistics | Console output: meals <50% accuracy ratio |
| **7** | Visualize meal metrics | 5 meal-related figures + statistics |
| **7.5** | Analyze FR1 meal accuracy distribution | Stacked histogram of high/low accuracy meals over time |

**FR1 Metrics Computed:**
- ✓ Overall ending accuracy
- ✓ Time to 80% learning milestone
- ✓ Average pellets per hour
- ✓ First meal latency
- ✓ First good meal latency (ML-classified)
- ✓ In-meal pellet ratio (organized vs scattered eating)
- ✓ Good meal proportion
- ✓ Low-accuracy meal ratio (meals <50% / total meals)

### 🔄 Part B: Reversal Learning Analysis (Steps 8-15.6)

| Step | Description | Key Outputs |
|------|-------------|-------------|
| **8** | Setup reversal parameters & pre-compute blocks/meals | `rev_session_analyses` with blocks + meals |
| **9** | Compute block transitions | Success rates, transition patterns, per-mouse plots |
| **10** | Visualize transition metrics | 4 group-level figures + t-tests |
| **11** | Compute learning scores | Early (75%) vs late (25%) block accuracy |
| **12** | Visualize learning dynamics | Score trends, result distributions, pellet ratios |
| **13** | Reversal block accuracy distribution | Match vs Mismatch FR1 active poke analysis |
| **14** | Compute retrieval time metrics | Mean, projected, and slope metrics per block |
| **15** | Analyze reversal meal patterns | 6 meal metrics during cognitive challenge |
| **15.5** | Calculate average meal size | Pellets per meal for reversal sessions |
| **15.6** | Meal accuracy vs dispense time correlation | Scatter plot with regression analysis |

**Reversal Metrics Computed:**
- ✓ Number of blocks per session
- ✓ First good meal time per block
- ✓ Meal accuracy during blocks
- ✓ Learning score (early adaptation, 0-75%)
- ✓ Learning result (late performance, 75-100%)
- ✓ Pellet-in-meal ratio trends
- ✓ Retrieval time dynamics (mean, slope, projection)
- ✓ Low-accuracy meal ratio
- ✓ Average meal size (pellets per meal)
- ✓ Meal accuracy vs dispense time correlation

### 📊 Part C: Inter-Pellet Interval Analysis (Steps 16-17)

| Step | Description | Key Outputs |
|------|-------------|-------------|
| **16** | Calculate inter-pellet intervals by position | `ipi_data_fr1`, `ipi_data_rev` |
| **17** | Visualize inter-pellet intervals | Per-group violin plots for pellet positions 2-12 |

### 📤 Part D: Data Export (Step 19)

| Step | Description | Key Outputs |
|------|-------------|-------------|
| **19** | Export all plotting data to Excel | Multi-sheet Excel file with all metrics |

---

## 🧠 Meal Quality Classification

The `Accurate Meal Model.ipynb` notebook provides a complete workflow for training neural network classifiers to distinguish high-quality feeding bouts from poor ones.

### Workflow Overview

```
1. Extract Meal Sequences → 2. K-means Clustering 
→ 3. Manual Selection for good (expected) clusters 
→ 4. Train LSTM/CNN → 5. Evaluate Performance → 6. Save Model Weights
```

### Methodology

**1. Meal Detection**
- Time threshold: ≤60 seconds between pellets
- Minimum pellets: ≥2 pellets per meal
- Extracts between-pellet accuracy sequences for each meal (e.g., `[100, 100, 50, 100]` means accuracy of a 5-pellet meal is 100%, 100%, 50%, 100% between each two pellets)

**2. Unsupervised Clustering**
- Uses K-means on meals grouped by pellet count (3, 4, 5+ pellets)
- Elbow method and Silhouette score helps you to estimate the optimal K

**3. Manual Annotation**
- Inspect cluster samples (accuracy patterns)
- Label clusters as "good" (consistent high accuracy or significant increasing trend on accuracy) or "bad" (inconsistent/low)
- Example: `[100, 100]` = good, `[50, 90, 100]` = good, `[100, 50]` = bad, `[50, 55, 60, 50]` = bad

**4. Model Training**
- **LSTM**: 2-layer RNN with 400 hidden units (~99% test accuracy)
- **CNN**: 1D convolutional network with dropout (~98% test accuracy)
- Binary classification: 0=good, 1=bad

**5. Deployment**
- Pre-trained models: `CNN_from_CASK.pth`, `LSTM_from_CASK.pth`
- Used automatically in main pipeline during meal analysis
- Retrain on your own data for experiment-specific classifiers

### Training Your Own Classifier (Light Computation)

If you have multiple experimental groups and want custom meal quality models:

1. Open `Accurate Meal Model.ipynb`
2. Run cells 1-3 to load your reversal sessions
3. For each group and pellet count (3, 4, 5):
   - Run elbow method to estimate K
   - Fit K-means and inspect cluster samples
   - Identify which clusters represent "good" meals
   - Save labeled data to `.pkl` files
4. Run training cells to train LSTM/CNN and evaluate
5. Save your model weights in notebook: `torch.save(model.state_dict(), 'data/CNN_from_YOUR_NAME.pth')`

Our LSTM model takes < 30s and CNN model takes < 10s to train on Apple M1 CPU, so you do not have to use a GPU! 

**Model Performance (CASK dataset):**
- LSTM: ≈99% test accuracy, F1≈0.99-1.0
- CNN: ≈98-99% test accuracy, F1≈0.98

---

## 🛠️ Script Modules Reference

All notebooks import reusable functions from `scripts/`. Here's a detailed breakdown:

### `preprocessing.py` - Data Loading & Quality Control

**Core Classes:**
- `SessionKey`: Metadata container (mouse_id, group, session_type, session_path)
- `SessionData`: Holds raw dataframe + computed key

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `build_session_catalog(sample_root, group_map_path)` | Scans data directory, loads all sessions, organizes by group/type |
| `session_cache(sample_root, group_map_path)` | Cached version of catalog builder (speeds up repeated runs) |
| `load_session_csv(csv_path)` | Reads FED3 CSV, cleans columns, adds accuracy calculations |
| `motor_turn_summary(csv_path, cutoff=15)` | Counts dispenser errors (motor turns ≥15), returns proportion |
| `calculate_accuracy_by_row(df)` | Computes row-wise accuracy from "Event" and "Active_Poke" columns |
| `get_retrieval_time(csv_path, day)` | Extracts pellet retrieval durations (poke → well entry) |
| `infer_session_type(session_df)` | Auto-detects FR1 vs REV based on active poke switches |

**Usage Example:**
```python
from scripts.preprocessing import build_session_catalog

SESSIONS, GROUPINGS = build_session_catalog('sample_data', 'group_map.json')
# SESSIONS: {session_id: SessionData}
# GROUPINGS: {group_name: {'FR1': [keys], 'REV': [keys]}}
```

---

### `accuracy.py` - Learning Curve Analysis

| Function | Purpose |
|----------|---------|
| `read_and_record(session, ending_corr, learned_time, acc_dict)` | Processes one session: computes final accuracy, 80% milestone, returns binned dataframe |
| `plot_cumulative_accuracy(dataframes, group_labels, bin_size_sec)` | Plots learning curves with SEM error bands across groups |
| `find_learning_milestone(data, window_hours, accuracy_threshold)` | Finds first timepoint when rolling accuracy ≥ threshold |
| `calculate_accuracy(group)` | Computes overall accuracy for a dataframe |
| `find_inactive_index(hourly_labels, rev)` | Detects inactive periods (used for block visualization) |

**Usage Example:**
```python
from scripts.accuracy import read_and_record, plot_cumulative_accuracy

ending_acc, learned_time = [], []
fr1_dfs = []
for session in fr1_sessions:
    df = read_and_record(session, ending_acc, learned_time, {})
    fr1_dfs.append(df)

plot_cumulative_accuracy([fr1_dfs], group_labels=['Control'], bin_size_sec=5)
```

---

### `meals.py` - Feeding Pattern Analysis

| Function | Purpose |
|----------|---------|
| `process_meal_data(session, export_root, prefix)` | **Main meal analysis function for FR1**: detects meals, classifies quality, computes 7+ metrics, generates plots |
| `process_meal_data_with_blocks(session, blocks, ...)` | **Block-aware meal analysis**: uses pre-computed blocks to ensure no cross-block meals |
| `find_meals_paper(data, time_threshold, pellet_threshold, method='paper')` | Detects meal boundaries using time-based clustering. Use `method='paper'` (default) or `method='ipi'` |
| `find_meals_by_blocks(blocks, ...)` | **Detects meals within each block separately**, ensuring meal boundaries respect block boundaries |
| `analyze_meals_by_blocks(blocks, ...)` | Combines block-based meal detection with ML quality classification |
| `predict_meal_quality(batch_meals, model_type)` | Runs LSTM/CNN classifier on meal sequences to predict good/bad |
| `find_first_accurate_meal(data, time_threshold, pellet_threshold)` | Finds first ML-classified "good" meal in session |
| `analyze_meals(data, meals, time_threshold, pellet_threshold)` | Batch-processes meals: computes stats, applies ML model |
| `calculate_low_accuracy_meal_ratio(data, ...)` | **Calculates ratio of meals <50% accuracy** (time/pellet constrained only, no accuracy filter) |
| `average_pellet(group)` | Calculates pellets per day |
| `pellet_flip(data)` | Aggregates pellet events into 10-minute bins |
| `active_meal(meals)` | Computes proportion of meals during active periods |
| `collect_good_meal_ratio(quality_map)` | Aggregates good/bad meal proportions across sessions |
| `graph_pellet_frequency(grouped_data, ...)` | Plots 10-minute pellet count histogram |
| `graphing_cum_count(data, meal, ...)` | Plots cumulative pellet curve with meal periods highlighted |

**Block-Based Meal Detection (Reversal Sessions):**
For reversal sessions, meals are detected within each block separately to ensure no cross-block meals. This guarantees consistency between transition analysis and meal metrics.

```python
from scripts.meals import find_meals_by_blocks, analyze_meals_by_blocks

# Detect meals within blocks (no cross-block meals)
session_meals, meal_acc, block_meal_info = find_meals_by_blocks(blocks)

# Or use the full analysis function
analysis = analyze_meals_by_blocks(blocks, method='ipi')
```

**Meal Metrics Returned by `process_meal_data`:**
- `avg_pellet`: Pellets per day
- `fir_meal`: First meal time (hours)
- `fir_good_meal`: First good meal time (hours)
- `inactive_meals`: Proportion of meals during inactive periods
- `in_meal_ratio`: Fraction of pellets inside meals
- `total_meals`: Number of meals detected
- `good_mask`: Boolean array of meal quality predictions
- `meal_count`: Meals per day
- `meals_with_acc`: List of [start_time, padded_accuracy_sequence] for each meal

**Metrics Returned by `calculate_low_accuracy_meal_ratio`:**
- `total_meals`: Total meals (constrained by time/pellet only, no accuracy filter)
- `low_accuracy_meals`: Number of meals with accuracy < cutoff
- `high_accuracy_meals`: Number of meals with accuracy ≥ cutoff
- `low_accuracy_ratio`: Fraction of low-accuracy meals
- `meal_accuracies`: List of all meal accuracy values

**Usage Example:**
```python
from scripts.meals import process_meal_data, calculate_low_accuracy_meal_ratio

# Main meal analysis
metrics = process_meal_data(session, export_root='figures/FR1/meals/')
print(f"Average pellets/day: {metrics['avg_pellet']}")
print(f"Good meal proportion: {sum(metrics['good_mask']) / metrics['total_meals']}")

# Calculate low-accuracy meal statistics (no accuracy filter applied)
stats = calculate_low_accuracy_meal_ratio(
    session.raw.copy(),
    time_threshold=60,
    pellet_threshold=2,
    accuracy_cutoff=50.0,
    method='paper',
)
print(f"Total meals (time/pellet only): {stats['total_meals']}")
print(f"Low accuracy meals (<50%): {stats['low_accuracy_meals']} ({stats['low_accuracy_ratio']*100:.1f}%)")
```

---

### `direction_transition.py` - Reversal Learning Analysis

| Function | Purpose |
|----------|---------|
| `split_data_to_blocks(data, day)` | **Splits reversal session into blocks** when active poke switches |
| `compute_session_analysis(data, day_limit, meal_config, method)` | **Main entry point**: computes blocks and meals together, returns all data for reuse |
| `get_transition_info(blocks, meal_config, reverse, block_meal_info, first_good_times)` | **Computes per-block stats**: transitions, success rate, meal timing. Optionally uses pre-computed meal data |
| `learning_score(blocks, block_prop, action_prop)` | **Early adaptation metric**: accuracy in first X% of each block |
| `learning_result(blocks, action_prop)` | **Late performance metric**: accuracy in last X% across all blocks |
| `first_meal_stats(data_stats, ignore_inactive)` | Extracts first meal ratio and timing from block stats |
| `plot_transition_stats(stats, blocks, ...)` | **Generates per-mouse transition plot**: bars + line plots + annotations |
| `plot_learning_score_trend(blocks_groups, ...)` | **Plots learning score curves** across action proportions |
| `plot_pellet_ratio_trend(blocks_groups, ...)` | **Plots in-meal pellet ratio trends** across blocks |
| `block_retrieval_summary(blocks, n_stds)` | Computes mean retrieval time per block, fits linear trend |
| `plot_retrieval_time_by_block(block_means, ...)` | Plots retrieval time trend with linear fit |
| `count_transitions(sub_frame)` | Counts L→L, L→R, R→L, R→R poke transitions |
| `find_inactive_blocks(blocks, reverse)` | Identifies blocks with minimal activity |
| `block_accuracy_by_proportion(blocks, proportion)` | Gets accuracy at specific percentage through each block |

**Efficient Reversal Analysis with `compute_session_analysis`:**
This is the recommended entry point for reversal session analysis. It computes blocks once and detects meals within each block, ensuring no cross-block meals and avoiding multiple recomputation of meals.

```python
from scripts.direction_transition import compute_session_analysis, get_transition_info

# Compute everything at once
analysis = compute_session_analysis(
    data=session.raw.copy(),
    day_limit=3,
    meal_config=(60, 2),
    method='ipi',
)

blocks = analysis['blocks']
meal_analysis = analysis['meal_analysis']
block_meal_info = meal_analysis['block_meal_info']

# Use pre-computed data for transition stats (no recomputation)
stats = get_transition_info(
    blocks, [60, 2], reverse=False,
    block_meal_info=block_meal_info,
    first_good_times=analysis['first_good_times_per_block'],
)
```

**Block Transition Patterns:**
- `L→L`: Repeated left pokes (perseveration)
- `L→R`: Left to right switch (exploration)
- `R→R`: Repeated right pokes
- `R→L`: Right to left switch

**Usage Example:**
```python
from scripts.direction_transition import split_data_to_blocks, learning_score

blocks = split_data_to_blocks(session.raw, day=3)
score = learning_score(blocks, block_prop=1.0, action_prop=0.75)
print(f"Learning score (0-75% of blocks): {score:.2%}")
```

---

### `utils.py` - Statistics & Visualization Helpers

| Function | Purpose |
|----------|---------|
| `perform_T_test(ctrl, exp, test_side, alpha, paired)` | Runs t-test, returns t-statistic, p-value, significance |
| `graph_group_stats(data_map, title, unit, ...)` | Creates violin plots with significance annotations |
| `run_pairwise_tests(metric_map, metric_name, cohort_pairs)` | Runs t-tests for all group pairs, prints results |
| `plot_group_stats_wrapper(...)` | **Convenience wrapper**: creates violin plot + outlier removal |
| `collect_metric(metric_name, mapping)` | Extracts specific metric from nested dictionary |

**Usage Example:**
```python
from scripts.utils import plot_group_stats_wrapper, run_pairwise_tests

# Plot with automatic outlier removal (>2.5 std)
plot_group_stats_wrapper(
    fr1_end_accuracy, 
    "Overall Accuracy", 
    "%", 
    "overall_accuracy.svg", 
    "figures/FR1", 
    remove_outlier_stds=2.5
)

# Statistical tests
TEST_PAIRS = [('control', 'experimental')]
run_pairwise_tests(fr1_end_accuracy, "Overall Accuracy", TEST_PAIRS)
```

---

### `meal_classifiers.py` - Neural Network Models

**Model Classes:**
- `RNNClassifier(input_size, hidden_size, num_layers, num_classes)`: 2-layer LSTM
- `CNNClassifier(num_classes, maxlen)`: 1D CNN with dropout
- `TimeSeriesDataset(X, y)`: PyTorch dataset wrapper

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `train(model, lr, num_epochs, train_loader, X_test, y_test)` | Trains model with Adam optimizer, prints progress |
| `evaluate_meals_by_groups(model, ctrl_input, ctrl_y, exp_input, exp_y)` | Evaluates accuracy, F1, good meal proportions for two groups |
| `evaluate_meals_on_new_data(model, ctrl_input, exp_input)` | Runs inference on new data without labels |
| `predict(model, input)` | Single prediction (0=good, 1=bad) |

**Usage Example:**
```python
from scripts.meal_classifiers import CNNClassifier
import torch

# Load pre-trained model
model = CNNClassifier(num_classes=2, maxlen=4)
model.load_state_dict(torch.load('data/CNN_from_CASK.pth'))
model.eval()

# Predict on new meal
meal_sequence = torch.tensor([[100, 100, -1, -1]], dtype=torch.float32)
with torch.no_grad():
    output = model(meal_sequence)
    prediction = torch.argmax(output, dim=1)  # 0=good, 1=bad
```

---

### `unsupervised_helpers.py` - Clustering & Data Prep

| Function | Purpose |
|----------|---------|
| `extract_meal_sequences(session_list, ...)` | Extracts accuracy sequences for all meals in sessions |
| `find_k_by_elbow(data)` | Plots elbow curve to estimate optimal K for K-means |
| `fit_model_single(data, k, visualize)` | Fits K-means, returns model + meals grouped by cluster |
| `collect_meals_from_categories(meals_by_category, good_class)` | Separates good/bad meals based on cluster labels |
| `data_padding(data)` | Pads variable-length sequences to fixed length (-1 padding) |
| `read_data(filename)` | Loads pickled meal data |
| `update_data(filename, new_list)` | Appends new meals to existing pickle file |

**Usage Example:**
```python
from scripts.unsupervised_helpers import extract_meal_sequences, find_k_by_elbow

sequences, good_ratios = extract_meal_sequences(rev_sessions)
three_pellet_meals = sequences.get(3, [])

# Find optimal K
find_k_by_elbow(three_pellet_meals)  # Displays elbow plot
```

---

### `advanced_analysis.py` - Specialized Analyses

This module provides advanced visualization and correlation analyses for deeper behavioral insights.

| Function | Purpose |
|----------|---------|
| `plot_fr1_meal_accuracy_distribution(fr1_group_sessions, bin_size_hours, export_path)` | **FR1 meal accuracy over time**: Stacked histogram showing frequency of high (≥50%) vs low (<50%) accuracy meals binned by hours from session start |
| `plot_reversal_block_accuracy_distribution(fr1_group_sessions, rev_group_sessions, export_root)` | **FR1 influence on reversal**: Compares meal accuracy distribution within blocks that match vs mismatch the FR1 active poke side. Also plots meal size distributions |
| `calculate_dispense_delays(csv_path, max_retrieval_gap, max_dispense_delay)` | **Mechanical timing analysis**: Estimates dispense delay (correct poke → pellet drop) by subtracting retrieval time from pellet timestamp |
| `plot_meal_dispense_time_correlation(rev_group_sessions, export_root)` | **Accuracy vs hardware**: Scatter plot with regression showing relationship between meal accuracy and average dispensing delay |

**Usage Example:**
```python
from scripts.advanced_analysis import (
    plot_fr1_meal_accuracy_distribution,
    plot_reversal_block_accuracy_distribution,
    plot_meal_dispense_time_correlation,
)

# FR1: Show meal accuracy distribution over 24 hours
plot_fr1_meal_accuracy_distribution(
    fr1_group_sessions,
    bin_size_hours=1,
    export_path='figures/FR1/fr1_meal_accuracy_distribution.svg'
)

# Reversal: Analyze FR1 influence on block performance
plot_reversal_block_accuracy_distribution(
    fr1_group_sessions,
    rev_group_sessions,
    export_root='figures/REV'
)

# Reversal: Correlate meal accuracy with mechanical delays
plot_meal_dispense_time_correlation(
    rev_group_sessions,
    export_root='figures/REV'
)
```

**Key Concepts:**

- **Match vs Mismatch blocks**: In reversal, blocks where the active poke matches the FR1 training side ("Match") vs blocks with the opposite side ("Mismatch"). This reveals transfer effects from initial learning.

- **Dispense delay**: Time between the correct active poke and when the pellet actually drops. Calculated as: `Pellet_Time - Retrieval_Time - Trigger_Poke_Time`. High delays may indicate mechanical issues.

- **Block progress**: Meals are positioned by their relative timing within a block (0-100%), allowing comparison of early vs late block feeding patterns.

---

## 📊 Output Files & Figures

Running the complete pipeline generates organized visualizations:

### FR1 Output (`figures/FR1/`)
```
fr1_cumulative_accuracy.svg           # Learning curves with SEM bands
fr1_overall_accuracy.svg              # Final accuracy distribution
fr1_learning_milestone_time.svg       # Start time to maintaining 80% accuracy for 2 hours
fr1_avg_pellets.svg                   # Pellet consumption rate
fr1_first_meal_time.svg               # Initial meal latency
fr1_first_good_meal_time.svg          # First quality meal latency
fr1_in_meal_ratio.svg                 # Organized vs scattered eating
fr1_good_meal_ratio.svg               # Proportion of quality meals
fr1_meal_accuracy_distribution.svg    # Stacked histogram: high vs low accuracy meals over time

meals/
├── control_M1_fr1_pellet_frequency.svg  # Per-session pellet frequency
├── control_M1_fr1_cumulative_sum.svg    # Per-session cumulative pellets
└── ...

interpellet_intervals/
├── fr1_ipi_control_pellets_2_12.svg  # FR1 feeding rhythm for Control group
├── fr1_ipi_cask_pellets_2_12.svg     # FR1 feeding rhythm for Cask group
└── ...
```

### Reversal Output (`figures/REV/`)
```
rev_learning_score_overall.svg        # Early adaptation curves
rev_learning_score.svg                # Learning score distribution
rev_learning_result.svg               # Final performance distribution
rev_pellet_ratio_overall.svg          # In-meal ratio trends
rev_number_of_blocks.svg              # Block count per session
rev_first_good_meal_time.svg          # Adaptation speed per block
rev_first_meal_ratio.svg              # Meal timing relative to block
rev_meal_accuracy.svg                 # Meal quality during reversals
rev_retrieval_mean.svg                # Average retrieval times
rev_retrieval_projection.svg          # Projected final retrieval
rev_retrieval_slope.svg               # Retrieval time trends
rev_avg_pellet.svg                    # Average pellets per day
rev_good_meal_ratio.svg               # Good meal proportion
rev_inactive_meals.svg                # Inactive meal fraction
rev_in_meal_ratio.svg                 # In-meal pellet ratio
rev_block_acc_dist_*.svg              # Block accuracy distribution (Match vs Mismatch FR1)
rev_block_meal_size_dist_*.svg        # Meal size distribution per block type
rev_acc_vs_dispense_delay_*.svg       # Meal accuracy vs dispensing delay correlation

transition/
├── control_M10_reversal_transition.svg  # Per-mouse block analysis
└── ...

retrieval/
├── control_M10_reversal_retrieval.svg   # Per-mouse retrieval trends
└── ...

meals/
├── control_M10_reversal_pellet_frequency.svg   # Per-session pellet frequency
├── control_M10_reversal_cumulative_sum.svg     # Per-session cumulative pellets
└── ...

interpellet_intervals/
├── rev_ipi_control_pellets_2_12.svg  # Reversal feeding rhythm for Control group
├── rev_ipi_cask_pellets_2_12.svg     # Reversal feeding rhythm for Cask group
└── ...
```

### Data Export Output (`figures/[method]/`)
```
[method]_analysis_data_export.xlsx    # Multi-sheet Excel with all plotting data
```
Each sheet contains raw values used to generate corresponding plots, organized by group.

All figures are publication-ready SVG format with:
- Clear axis labels and units
- Group color coding
- Statistical significance annotations
- Error bands (SEM or std)

---

## ⚙️ Customization Guide

### Modify Analysis Parameters

Edit these values directly in `pipeline.ipynb`:

```python
# Step 6: FR1 Meal Detection
time_threshold = 60      # Maximum seconds between pellets in a meal
pellet_threshold = 2     # Minimum pellets required for meal

# Step 8: Reversal Analysis
REV_DAY_LIMIT = 3        # Analyze only first 3 days of reversal data
REV_MEAL_CONFIG = (60, 2)  # (time_threshold, pellet_threshold)

# Step 5, 10, 14, etc.: Outlier Removal
remove_outlier_stds = 2.5  # Remove values >2.5 std from mean in violin-box plots
```

### Add New Experimental Groups

1. Update `group_map.json`:
```json
{
  "control": ["M1", "M2"],
  "new_group": ["M50", "M51", "M52"]
}
```

2. Re-run notebook - groups are auto-detected!

3. (Optional) Update test pairs for specific comparisons (You can enter more than one pairs):
```python
TEST_PAIRS = [
    ('control', 'new_group'),
    ('experimental', 'new_group')
]
```

### Use Custom Meal Classifier

Train your own model in `Accurate Meal Model.ipynb`, then update model loading in `scripts/meals.py`:

```python
def _build_meal_model(model_type: str):
    if model_type == 'cnn':
        model = CNNClassifier(num_classes=2, maxlen=4)
        model.load_state_dict(torch.load('data/CNN_from_YOUR_NAME.pth'))
    # ...
```

### Extend with Custom Metrics

Add new analysis to appropriate script module:

```python
# scripts/custom_analysis.py
def my_custom_metric(session_data):
    """Compute custom behavioral metric."""
    df = session_data.raw
    # Your analysis here
    return result
```

Then import and use in notebook:
```python
from scripts.custom_analysis import my_custom_metric

custom_results = {group: [] for group in GROUPS}
for group, sessions in GROUP_SESSIONS.items():
    for session in sessions:
        metric = my_custom_metric(session)
        custom_results[group].append(metric)

plot_group_stats_wrapper(custom_results, "My Metric", "units", "custom.svg", "figures/")
```

---

## 🐛 Troubleshooting

| Issue | Possible Causes | Solution |
|-------|----------------|----------|
| **"No sessions found"** | Incorrect data structure | Ensure CSVs are in `DATA_DIR/*/` format and remember updating your directory of data in notebook |
| **Missing meal classifier** | Model file not present | Check `data/CNN_from_CASK.pth` exists, or train your own |
| **Import errors** | Missing dependencies | Run `pip install -r requirements.txt` |
| **High memory usage** | Large cached data | Call `session_cache.cache_clear()` in Step 2 |
| **Empty reversal results** | No REV sessions in data | Ensure mice have `reversal.csv` files |
| **Step 3 has non-empty data frame output** | Hardware malfunctions | Check removed_sessions table; auto-filtered if >20% errors already |

### Common Data Issues

**CSV Format Requirements:**
- Must have columns: `MM:DD:YYYY hh:mm:ss`, `Event`, `Active_Poke`, `Left_Poke_Count`, `Right_Poke_Count`
- Event types: "Left", "Right", "Pellet", "LeftWithDispense" and other "WithXXX" item start with "Left" or "Right"
- Active_Poke values: "Left", "Right"

**Group Map Issues:**
If you get "KeyError: mouse_id", ensure:
1. `group_map.json` includes ALL mice in `sample_data/`
2. Mouse folder names match exactly (case-sensitive)
3. JSON is valid (use [jsonlint.com](https://jsonlint.com/))

---