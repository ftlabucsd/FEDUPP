# Script Observations

## `scripts/preprocessing.py`
- Central entry-point for CSV ingestion; builds cached `SessionData` objects from `sample_data/` and `group_map.json`.
- Provides group/session accessors (`session_cache`, `get_group_sessions`, `get_session_data`) and legacy helpers (`calculate_accuracy_by_row`).
- Normalizes timestamps, accuracy/preference columns, and safeguards around missing/bad rows.

## `scripts/accuracy.py`
- Operates on in-memory `SessionData` frames and offers cumulative accuracy summarisation/plotting utilities.
- `read_and_record` now accepts a `SessionData`, avoiding redundant disk IO.
- Plotting helpers remain reusable; exporting path passed in explicitly.

## `scripts/meals.py`
- Meal metrics consume `SessionData`; `process_meal_data` now handles session metadata for naming/exports.
- Still includes classical plotting functions and neural helpers; further modularisation possible later.

## `scripts/intervals.py`
- Will require follow-up refactor to use session cache instead of Excel file path inputs.

## `scripts/direction_transition.py`
- Continues to rely on legacy path parsing; not yet migrated to cached sessions.

## `scripts/meal_classifiers.py`
- Self-contained PyTorch models; unchanged in this pass.

## `scripts/unsupervised_helpers.py`
- Still expects Excel ingestion; slated for future parity update.

