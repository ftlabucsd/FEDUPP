# FEDUPP - FED3 Users Processing Package

This is the official implementation of the paper "The development of FEDUPP: Feeding Experimentation Device Users Processing Package to Assess Learning and Cognitive Flexibility"

 [![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/) &nbsp; &nbsp; [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) &nbsp; &nbsp; [![Paper Link](https://img.shields.io/badge/bioarxiv-paper-red.svg)](https://www.biorxiv.org/content/10.1101/2025.08.14.670424v1)

> **A comprehensive, reusable analysis pipeline for FED3 behavioral data to assess learning acquisition, cognitive flexibility, and feeding patterns in mice.**

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
- Meal quality during cognitive challenge

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ftlabucsd/FEDUPP
cd FEDUPP

# Install dependencies (requires Python ≥3.10)
pip install -r requirements.txt
```

### 2. Data Preparation

**A. Organize Your Data**

Place your FED3 CSV files in `sample_data/` with this structure:

```
sample_data/
├── M1/
│   ├── fr1.csv
│   └── reversal.csv
├── M2/
│   └── fr1.csv
├── M3/
│   ├── fr1.csv
│   └── reversal.csv
└── ...
```
> Note: you do not have to specify the FR1 or reversal session type in csv filenames. Our algorithm will automatically determine its session. The names is for illustration only.

**B. Define Group Membership**

Create or modify `group_map.json` to assign mice id to experimental groups, for example like below (the ID you enter here must match the subfolder name, like "M1", "M2" above):

```json
{
  "control": ["M1", "M2", "M3"],
  "experimental": ["M10", "M11", "M12"],
  "validation": ["M20", "M21", "M22"]
}
```

### 3. Run the Analysis Pipeline

Inside this project, open `pipeline.ipynb` in Jupyter Lab or VS Code or other IDEs and run cells sequentially:

The notebook will:
1. Load and validate your data
2. Perform quality control checks
3. Generate FR1 and Reversal Learning analyses
4. Save figures to `figures/FR1/` and `figures/REV/`
5. Print statistical test results

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
└── scripts/                   # Python modules (imported by notebooks)
    ├── preprocessing.py
    ├── accuracy.py
    ├── meals.py
    ├── direction_transition.py
    ├── utils.py
    ├── meal_classifiers.py
    └── unsupervised_helpers.py
```

---

## 📓 Pipeline Notebook Guide

The `pipeline_empty.ipynb` is organized into **15 sequential steps** across two main sections:

### 🔧 Setup & Quality Control (Steps 1-3)

| Step | Description | Output |
|------|-------------|--------|
| **1** | Import libraries and helper functions | Ready environment |
| **2** | Load session catalog and group assignments | `SESSIONS`, `GROUPINGS` dictionaries |
| **3** | Check dispenser motor performance | Remove sessions with >20% mechanical errors |

### 📈 Part A: FR1 Analysis (Steps 4-7)

| Step | Description | Key Outputs |
|------|-------------|-------------|
| **4** | Compute learning metrics | `fr1_overall_accuracy`, `fr1_learning_milestone` |
| **5** | Visualize FR1 performance | Accuracy & milestone plots + t-tests |
| **6** | Analyze meal patterns | Pellet rates, meal timing, quality metrics |
| **7** | Visualize meal metrics | 5 meal-related figures + statistics |

**FR1 Metrics Computed:**
- ✓ Overall ending accuracy
- ✓ Time to 80% learning milestone
- ✓ Average pellets per hour
- ✓ First meal latency
- ✓ First good meal latency (ML-classified)
- ✓ In-meal pellet ratio (organized vs scattered eating)
- ✓ Good meal proportion

### 🔄 Part B: Reversal Learning Analysis (Steps 8-15)

| Step | Description | Key Outputs |
|------|-------------|-------------|
| **8** | Setup reversal parameters | Filter REV sessions, set day limits |
| **9** | Compute block transitions | Success rates, transition patterns, per-mouse plots |
| **10** | Visualize transition metrics | 4 group-level figures + t-tests |
| **11** | Compute learning scores | Early (75%) vs late (25%) block accuracy |
| **12** | Visualize learning dynamics | Score trends, result distributions, pellet ratios |
| **13** | Analyze retrieval times | Mean, projected, and slope metrics per block |
| **14** | Visualize retrieval metrics | 3 retrieval figures + statistics |
| **15** | Analyze reversal meal patterns | 6 meal metrics during cognitive challenge |

**Reversal Metrics Computed:**
- ✓ Number of blocks per session
- ✓ First good meal time per block
- ✓ Meal accuracy during blocks
- ✓ Learning score (early adaptation, 0-75%)
- ✓ Learning result (late performance, 75-100%)
- ✓ Pellet-in-meal ratio trends
- ✓ Retrieval time dynamics (mean, slope, projection)

---

## 🧠 Meal Quality Classification

The `Accurate Meal Model.ipynb` notebook provides a complete workflow for training neural network classifiers to distinguish high-quality feeding bouts from poor ones.

### Workflow Overview

```
1. Extract Meal Sequences → 2. K-means Clustering → 3. Manual Selection for good (expected) clusters → 
4. Train LSTM/CNN → 5. Evaluate Performance → 6. Save Model Weights
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

### Training Your Own Classifier

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

**Model Performance (CASK dataset):**
- LSTM: ~99% test accuracy, F1~=0.99-1.0
- CNN: ~98-99% test accuracy, F1~=0.98

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
| `process_meal_data(session, export_root, prefix)` | **Main meal analysis function**: detects meals, classifies quality, computes 7+ metrics, generates plots |
| `find_meals_paper(data, time_threshold, pellet_threshold)` | Detects meal boundaries using time-based clustering |
| `predict_meal_quality(batch_meals, model_type)` | Runs LSTM/CNN classifier on meal sequences to predict good/bad |
| `find_first_accurate_meal(data, time_threshold, pellet_threshold)` | Finds first ML-classified "good" meal in session |
| `analyze_meals(data, meals, time_threshold, pellet_threshold)` | Batch-processes meals: computes stats, applies ML model |
| `average_pellet(group)` | Calculates pellets per hour |
| `pellet_flip(data)` | Adjusts poke counts during reversal blocks |
| `active_meal(meals)` | Computes proportion of meals during active periods |
| `collect_good_meal_ratio(quality_map)` | Aggregates good/bad meal proportions across sessions |
| `graph_pellet_frequency(grouped_data, ...)` | Plots inter-pellet interval histogram |
| `graphing_cum_count(data, meal, ...)` | Plots cumulative pellet curve with meal periods highlighted |

**Meal Metrics Returned by `process_meal_data`:**
- `avg_pellet`: Pellets per hour
- `fir_meal`: First meal time (hours)
- `fir_good_meal`: First good meal time (hours)
- `inactive_meals`: Proportion of meals during inactive periods
- `in_meal_ratio`: Fraction of pellets inside meals
- `total_meals`: Number of meals detected
- `good_mask`: Boolean array of meal quality predictions

**Usage Example:**
```python
from scripts.meals import process_meal_data

metrics = process_meal_data(session, export_root='figures/FR1/meals/')
print(f"Average pellets/hour: {metrics['avg_pellet']}")
print(f"Good meal proportion: {sum(metrics['good_mask']) / metrics['total_meals']}")
```

---

### `direction_transition.py` - Reversal Learning Analysis

| Function | Purpose |
|----------|---------|
| `split_data_to_blocks(data, day)` | **Splits reversal session into blocks** when active poke switches |
| `get_transition_info(blocks, meal_config, reverse)` | **Computes per-block stats**: transitions, success rate, meal timing |
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

## 📊 Output Files & Figures

Running the complete pipeline generates organized visualizations:

### FR1 Output (`figures/FR1/`)
```
cumulative_accuracy.svg          # Learning curves with SEM bands
overall_accuracy.svg              # Final accuracy distribution
learning_milestone_time.svg       # Time to 80% accuracy
avg_pellets.svg                   # Pellet consumption rate
first_meal_time.svg               # Initial meal latency
first_good_meal_time.svg          # First quality meal latency
in_meal_ratio.svg                 # Organized vs scattered eating
good_meal_ratio.svg               # Proportion of quality meals

meals/
├── control_M1_fr1_frequency.svg  # Per-session meal diagnostics
├── control_M1_fr1_cumulative.svg
└── ...
```

### Reversal Output (`figures/REV/`)
```
rev_learning_score_overall.svg       # Early adaptation curves
rev_learning_result.svg               # Final performance distribution
rev_pellet_ratio_overall.svg          # In-meal ratio trends
rev_number_of_blocks.svg              # Block count per session
rev_first_good_meal_time.svg          # Adaptation speed per block
rev_first_meal_ratio.svg              # Meal timing relative to block
rev_meal_accuracy.svg                 # Meal quality during reversals
rev_retrieval_mean.svg                # Average retrieval times
rev_retrieval_projection.svg          # Projected final retrieval
rev_retrieval_slope.svg               # Retrieval time trends

transition/
├── control_M10_reversal_transition.svg  # Per-mouse block analysis
└── ...

retrieval/
├── control_M10_reversal_retrieval.svg   # Per-mouse retrieval trends
└── ...

meals/
├── control_M10_reversal_frequency.svg   # Per-session meal diagnostics
└── ...
```

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
| **"No sessions found"** | Incorrect data structure | Ensure CSVs are in `sample_data/M*/` format |
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

## 📖 Citation

If you use FEDUPP in your research, please cite:

```bibtex
@article{FEDUPP,
  author  = {Mingyang Yao and Avraham M. Libster and Shane Desfor and Freiya Malhotra and Nathalia Castorena and Patricia Montilla-Perez and Francesca Telese},
  title   = {The development of FEDUPP: Feeding Experimentation Device Users Processing Package to Assess Learning and Cognitive Flexibility},
  year    = {2025},
  journal = {bioRxiv},
  url     = {https://www.biorxiv.org/content/early/2025/08/20/2025.08.14.670424},
  note    = {Mingyang Yao and Avraham M. Libster contributed equally.}
}
```

---

## 🤝 Contributing

We welcome contributions! To contribute:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**:
   - Add new metrics to appropriate script module
   - Update notebook with usage examples
   - Add docstrings to let users quickly know the input and expected output
4. **Test thoroughly** with sample data
5. **Submit a pull request** with clear description

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to function signatures
- Document all public functions with docstrings
- Test on Python 3.10+ environments
- Keep notebooks cell-by-cell executable

---

## 📧 Contact & Support

**Maintainer**: [FT Lab](https://www.teleselab.com/)
**Report Issues**: [GitHub Issues](https://github.com/your-username/FED3-data/issues)  

For bug reports, include:
- Error message / traceback
- Sample data or file that triggers the error (if possible)

---

## 📜 License

© 2025 FT Lab

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔄 Changelog

### v2.0 (January 2025)
- ✨ Complete pipeline reorganization with modular scripts
- 📖 Comprehensive inline documentation in notebooks
- 🔧 Moved all functions to `scripts/` for reusability
- 🎨 Improved visualization consistency (SVG outputs)
- 🧠 Added ML-based meal quality classification
- 📊 Enhanced reversal learning analysis (block transitions, retrieval times)
- 🧪 Automated quality control checks
- 📈 Statistical testing built into workflow

### v1.0 (2024)
- Initial release with CASK experiment analysis
- Basic FR1 and reversal learning support

---

## 🙏 Acknowledgments

- **FED3 Device**: [Kravitz Lab Open Source Hardware](https://github.com/KravitzLabDevices/FED3)
- **Community Contributors**: Thanks to all researchers who provided feedback and suggestions

---

## 📚 Additional Resources

- [FED3 Hardware Documentation](https://github.com/KravitzLabDevices/FED3)
- [FED3 User Guide](https://github.com/KravitzLabDevices/FED3/wiki)

---

**⭐ If this project helps your research, please give it a star on GitHub!**

