"""Common plotting utilities and statistical helpers for FED3 analyses."""
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from pathlib import Path

# Blue, Orange, Green, Red, Purple - You can extend this list for more groups.
palette = ['#425df5', '#f55442', '#0ec72a', '#f5e142', '#6b1cd9']


def perform_T_test(ctrl:list, exp:list, test_side='two-sided', alpha=0.05, paired=False):
    """Perform a t-test between control and experimental measurements."""
    if test_side not in ['two-sided', 'less', 'greater']:
        print('Test size must be two-sided, less or greater')
        return
    
    if paired:
        _, p_value = stats.ttest_rel(exp, ctrl, alternative=test_side)
    else:
        _, p_value = stats.ttest_ind(exp, ctrl, alternative=test_side)

    print("P Value is ", p_value)
    

def graph_group_stats(
    group_data: list,
    stats_name: str,
    unit: str,
    group_names: list | None = None,
    violin_width: float = 0.25,
    dpi: int = 150,
    verbose: bool = True,
    export_path: str | None = None,
    remove_outlier_stds: float = -1, # -1 means no outlier removal
):
    """Visualise summary statistics for one or more groups.

    Creates violin plots with inset boxplots and jittered scatter points for each
    group, optionally exporting the figure. Supports between 1 and 5 groups.

    Args:
        group_data (list[list[float]]): Sequence of observations per group.
        stats_name (str): Display name of the statistic (e.g., "Accuracy").
        unit (str): Unit label to append to the y-axis (e.g., "%").
        group_names (list[str], optional): Names for each group. Defaults to
            generated numeric labels when omitted.
        violin_width (float, optional): Width of each violin. Defaults to 0.25.
        dpi (int, optional): Figure DPI. Defaults to 150.
        verbose (bool, optional): When True, print summary statistics. Defaults to True.
        export_path (str, optional): When provided, save the figure to this path.
        remove_outlier_stds (float, optional): The number of standard deviations to use for outlier removal. Defaults to 2.5.
    """
    if not group_data:
        raise ValueError("group_data must contain at least one group.")

    n_groups = len(group_data)
    if group_names is None:
        group_names = [f"Group {idx+1}" for idx in range(n_groups)]
    if len(group_names) != n_groups:
        raise ValueError("group_names length must match group_data length.")

    prepared = []
    for idx, values in enumerate(group_data):
        # remove values exceeding certain number of std from the mean
        if remove_outlier_stds > 0:
            mean = np.mean(values)
            std = np.std(values)
            values = [value for value in values if value < mean + remove_outlier_stds * std and value > mean - remove_outlier_stds * std]
        if len(values) == 0:
            raise ValueError(f"Group '{group_names[idx]}' has no observations.")
        prepared.append(np.asarray(values, dtype=float))

    if verbose:
        for name, values in zip(group_names, prepared):
            mean_val = float(np.mean(values))
            se_val = float(np.std(values, ddof=0) / np.sqrt(len(values)))
            print(f"{name} Size: {len(values)} \t Average: {mean_val:.3f} \t SE: {se_val:.3f}")

    fig_width = max(6, 2 + 1.6 * n_groups)
    fig, ax = plt.subplots(dpi=dpi)
    fig.set_size_inches(fig_width, 6)

    x_positions = np.arange(n_groups)

    parts = ax.violinplot(
        prepared,
        positions=x_positions,
        widths=violin_width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, violin in enumerate(parts['bodies']):
        color = palette[i % len(palette)]
        violin.set_facecolor(color)
        violin.set_edgecolor('black')
        violin.set_alpha(0.65)

    ax.boxplot(
        prepared,
        positions=x_positions,
        widths=violin_width * 0.6,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor='white', edgecolor='black'),
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
    )

    jitter_strength = violin_width / 6
    for i, (x, values) in enumerate(zip(x_positions, prepared)):
        jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(values))
        ax.scatter(
            np.repeat(x, len(values)) + jitter,
            values,
            color=palette[i % len(palette)],
            edgecolor='black',
            linewidth=0.4,
            alpha=0.85,
            zorder=3,
        )

    legend_handles = [
        mpatches.Patch(color=palette[i % len(palette)], alpha=0.65, label=f"{name} (n={len(values)})")
        for i, (name, values) in enumerate(zip(group_names, prepared))
    ]
    ax.legend(handles=legend_handles, fontsize=12)

    ax.set_xlabel('Groups', fontsize=14)
    ax.set_ylabel(f"{stats_name} ({unit})", fontsize=14)
    ax.set_title(f"{stats_name} Distribution", fontsize=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_names, rotation=0)
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()


def run_pairwise_tests(metric_map: dict, metric_name: str, cohort_pairs: list | None = None):
    """Run independent t-tests for each specified pair of cohorts.
    
    Args:
        metric_map (dict): Dictionary mapping group names to lists of metric values.
        metric_name (str): Name of the metric being tested (for display).
        cohort_pairs (list, optional): List of tuples specifying which pairs to test.
            Each tuple should contain two group names. Defaults to None.
    
    Raises:
        KeyError: If requested group names are not found in metric_map.
    """
    cohort_pairs = cohort_pairs or []
    for g1, g2 in cohort_pairs:
        if g1 not in metric_map or g2 not in metric_map:
            raise KeyError(f"Requested cohorts {g1}, {g2} not found")
        series1, series2 = metric_map[g1], metric_map[g2]
        print(f"[T-test] {metric_name}: {g1} vs {g2}")
        perform_T_test(series1, series2, test_side="two-sided")


def plot_group_stats_wrapper(
    metric_map: dict,
    metric_name: str,
    unit: str,
    export_filename: str | None = None,
    figure_dir: str | None = None,
    remove_outlier_stds: float = -1
):
    """Convenience wrapper around graph_group_stats for dictionary input.
    
    Args:
        metric_map (dict): Dictionary mapping group names to lists of values.
        metric_name (str): Name of the metric (e.g., "Overall Accuracy").
        unit (str): Unit label for the y-axis (e.g., "%").
        export_filename (str, optional): Filename to save the figure. Requires figure_dir.
        figure_dir (str, optional): Directory path to save the figure.
        remove_outlier_stds (float, optional): Number of standard deviations for outlier removal.
            Defaults to -1 (no removal).
    
    Raises:
        ValueError: If export_filename is provided without figure_dir.
    """
    group_names = list(metric_map.keys())
    datasets = [metric_map[name] for name in group_names]
    if export_filename and figure_dir is None:
        raise ValueError("figure_dir must be provided when export_filename is set")
    
    export_path = str(Path(figure_dir) / export_filename) if export_filename else None
    
    graph_group_stats(
        group_data=datasets,
        stats_name=metric_name,
        unit=unit,
        group_names=group_names,
        export_path=export_path,
        remove_outlier_stds=remove_outlier_stds,
    )


def collect_metric(metric_name: str, mapping: dict) -> dict:
    """Extract a specific metric from nested dictionaries of results.
    
    Args:
        metric_name (str): Key to extract from each entry in the mapping values.
        mapping (dict): Dictionary where each value is a list of dictionaries.
    
    Returns:
        dict: Dictionary with the same keys, but values are lists of extracted metrics.
    """
    return {
        group: [entry[metric_name] for entry in metrics]
        for group, metrics in mapping.items()
    }