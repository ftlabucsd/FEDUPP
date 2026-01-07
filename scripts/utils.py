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
    """Visualise summary statistics for one or more groups in two subplots.

    Creates violin plots with inset boxplots and jittered scatter points for each
    group, optionally exporting the figure. Groups 'ctrl' and 'cask' are shown in 
    the left subplot, and all remaining groups are shown in the right subplot.

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
        remove_outlier_stds (float, optional): The number of standard deviations to use for outlier removal. Defaults to -1 (no removal).
    """
    if not group_data:
        raise ValueError("group_data must contain at least one group.")

    n_groups = len(group_data)
    if group_names is None:
        group_names = [f"Group {idx+1}" for idx in range(n_groups)]
    if len(group_names) != n_groups:
        raise ValueError("group_names length must match group_data length.")

    # Prepare data and remove outliers
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

    # Separate groups into two categories
    left_groups = []  # ctrl and cask
    left_names = []
    left_colors = []
    left_data = []
    
    right_groups = []  # remaining groups
    right_names = []
    right_colors = []
    right_data = []
    
    for idx, (name, values) in enumerate(zip(group_names, prepared)):
        color = palette[idx % len(palette)]
        if name.lower() in ['ctrl', 'cask']:
            left_groups.append(idx)
            left_names.append(name)
            left_colors.append(color)
            left_data.append(values)
        else:
            right_groups.append(idx)
            right_names.append(name)
            right_colors.append(color)
            right_data.append(values)
    
    # Calculate shared y-axis limits based on all data
    all_values = np.concatenate(prepared)
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    y_range = y_max - y_min
    
    # Add padding (10% on each side)
    padding = y_range * 0.1
    y_floor = y_min - padding
    y_ceil = y_max + padding
    
    # Calculate nice tick interval
    # Aim for approximately 5-8 major ticks
    rough_interval = y_range / 6
    # Round to a nice number
    magnitude = 10 ** np.floor(np.log10(rough_interval))
    normalized = rough_interval / magnitude
    if normalized < 1.5:
        nice_interval = magnitude
    elif normalized < 3:
        nice_interval = 2 * magnitude
    elif normalized < 7:
        nice_interval = 5 * magnitude
    else:
        nice_interval = 10 * magnitude
    
    # Shared hyperparameters for consistency
    shared_violin_width = violin_width
    shared_box_width = violin_width * 0.6
    shared_jitter_strength = violin_width / 6
    shared_alpha_violin = 0.65
    shared_alpha_scatter = 0.85
    shared_scatter_linewidth = 0.4
    
    # Create figure with two subplots
    fig, (ax_left, ax_right) = plt.subplots(1, 2, dpi=dpi)
    fig_width = max(12, 6 + 1.6 * n_groups)
    fig.set_size_inches(fig_width, 6)
    
    # Helper function to plot on a specific axis
    def plot_on_axis(ax, data, names, colors, title_suffix):
        if len(data) == 0:
            ax.set_visible(False)
            return
        
        n_groups_subplot = len(data)
        x_positions = np.arange(n_groups_subplot)
        
        # Violin plot (using shared hyperparameters)
        parts = ax.violinplot(
            data,
            positions=x_positions,
            widths=shared_violin_width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for i, violin in enumerate(parts['bodies']):
            violin.set_facecolor(colors[i])
            violin.set_edgecolor('black')
            violin.set_alpha(shared_alpha_violin)
        
        # Boxplot (using shared hyperparameters)
        ax.boxplot(
            data,
            positions=x_positions,
            widths=shared_box_width,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor='white', edgecolor='black'),
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
        )
        
        # Scatter points with jitter (using shared hyperparameters)
        for i, (x, values) in enumerate(zip(x_positions, data)):
            jitter = np.random.uniform(-shared_jitter_strength, shared_jitter_strength, size=len(values))
            ax.scatter(
                np.repeat(x, len(values)) + jitter,
                values,
                color=colors[i],
                edgecolor='black',
                linewidth=shared_scatter_linewidth,
                alpha=shared_alpha_scatter,
                zorder=3,
            )
        
        # Legend (using shared hyperparameters)
        legend_handles = [
            mpatches.Patch(color=colors[i], alpha=shared_alpha_violin, label=f"{name} (n={len(values)})")
            for i, (name, values) in enumerate(zip(names, data))
        ]
        ax.legend(handles=legend_handles, fontsize=12)
        
        # Labels and formatting
        ax.set_xlabel('Groups', fontsize=14)
        ax.set_ylabel(f"{stats_name} ({unit})", fontsize=14)
        ax.set_title(f"{stats_name} Distribution - {title_suffix}", fontsize=16)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(names, rotation=0)
        ax.set_xlim(-0.5, n_groups_subplot - 0.5)
        
        # Apply shared y-axis limits and ticks
        ax.set_ylim(y_floor, y_ceil)
        y_ticks = np.arange(
            np.ceil(y_floor / nice_interval) * nice_interval,
            y_ceil,
            nice_interval
        )
        ax.set_yticks(y_ticks)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Plot left subplot (ctrl and cask)
    plot_on_axis(ax_left, left_data, left_names, left_colors, "Control Groups")
    
    # Plot right subplot (remaining groups)
    plot_on_axis(ax_right, right_data, right_names, right_colors, "Other Groups")
    
    plt.tight_layout()
    
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


def calculate_interpellet_intervals_by_position(group_sessions: dict, method: str = 'ipi') -> dict:
    """Calculate inter-pellet intervals organized by pellet position within meals.
    
    Args:
        group_sessions (dict): Dictionary mapping group names to lists of SessionData objects.
        method (str): Meal detection method ('ipi' or 'paper'). Defaults to 'ipi'.
    
    Returns:
        dict: Nested dictionary where:
            - First level keys are pellet positions (2, 3, 4, ...)
            - Second level keys are group names
            - Values are lists of time intervals in seconds
    
    Example:
        {
            2: {'ctrl': [10.5, 12.3, ...], 'exp': [8.2, 9.1, ...]},
            3: {'ctrl': [11.2, 13.5, ...], 'exp': [9.5, 10.8, ...]},
            ...
        }
    """
    import pandas as pd
    from scripts.meals import find_meals_paper
    
    # Initialize data structure: pellet_position -> group -> list of intervals
    intervals_by_position = {}
    
    for group, sessions in group_sessions.items():
        for session in sessions:
            data = session.raw.copy()
            
            # Detect meals using the specified method
            meals, _ = find_meals_paper(
                data, 
                time_threshold=60, 
                pellet_threshold=2,
                in_meal_ratio=False,
                method=method
            )
            
            if not meals:
                continue
            
            # Get pellet events with retrieval timestamps
            df_pellets = data[data['Event'] == 'Pellet'].copy()
            # df_pellets['retrieval_timestamp'] = df_pellets['Time'] + pd.to_timedelta(
            #     df_pellets['collect_time'], unit='m'
            # )
            
            # For each meal, calculate inter-pellet intervals
            for meal_start, meal_end in meals:
                # Find pellets within this meal
                meal_pellets = df_pellets[
                    (df_pellets['Time'] >= meal_start) &
                    (df_pellets['Time'] <= meal_end)
                ].sort_values('Time')
                
                if len(meal_pellets) < 2:
                    continue
                
                # Calculate intervals between consecutive pellets
                timestamps = meal_pellets['Time'].values
                for i in range(1, len(timestamps)):
                    # Pellet position (2 = interval from 1st to 2nd pellet)
                    pellet_position = i + 1
                    
                    # Time interval in seconds
                    interval_seconds = (
                        pd.Timestamp(timestamps[i]) - pd.Timestamp(timestamps[i-1])
                    ).total_seconds()
                    
                    # Initialize nested structure if needed
                    if pellet_position not in intervals_by_position:
                        intervals_by_position[pellet_position] = {}
                    if group not in intervals_by_position[pellet_position]:
                        intervals_by_position[pellet_position][group] = []
                    
                    # Store the interval
                    intervals_by_position[pellet_position][group].append(interval_seconds)
    
    return intervals_by_position


def plot_interpellet_intervals_by_group_separate(
    intervals_by_position: dict,
    group_name: str,
    pellet_positions: list | None = None,
    export_path: str | None = None,
    dpi: int = 150,
):
    """Plot inter-pellet intervals as violin plots for each pellet position for a specific group.
    
    Args:
        intervals_by_position (dict): Output from calculate_interpellet_intervals_by_position.
        group_name (str): Name of the group to plot.
        pellet_positions (list, optional): Which pellet positions to plot. If None, plots all.
        export_path (str, optional): Path to save the figure.
        dpi (int): Figure DPI.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    if pellet_positions is None:
        pellet_positions = sorted(intervals_by_position.keys())
    
    # Create figure
    n_positions = len(pellet_positions)
    fig_width = max(12, 2 + 1.2 * n_positions)
    fig, ax = plt.subplots(dpi=dpi, figsize=(fig_width, 6))
    
    # Prepare data for each pellet position for this specific group
    positions_data = []
    positions_labels = []
    positions_counts = []
    
    for pos in pellet_positions:
        if pos not in intervals_by_position:
            continue
        
        # Get data for this group only
        if group_name in intervals_by_position[pos]:
            group_data = intervals_by_position[pos][group_name]
            if group_data:
                positions_data.append(np.array(group_data))
                positions_labels.append(str(pos))
                positions_counts.append(len(group_data))
    
    if not positions_data:
        print("No data remaining after filtering.")
        return
    
    x_positions = np.arange(len(positions_data))
    
    # Violin plot
    violin_width = 0.4
    parts = ax.violinplot(
        positions_data,
        positions=x_positions,
        widths=violin_width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for violin in parts['bodies']:
        violin.set_facecolor(palette[0])
        violin.set_edgecolor('black')
        violin.set_alpha(0.65)
    
    # Boxplot overlay
    ax.boxplot(
        positions_data,
        positions=x_positions,
        widths=violin_width * 0.6,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor='white', edgecolor='black'),
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
    )
    
    # Add N labels above each violin (removed scatter points for clarity)
    y_max = max(np.max(data) for data in positions_data)
    y_min = min(np.min(data) for data in positions_data)
    y_range = y_max - y_min
    
    # Position N labels with sufficient spacing from the data
    label_y_position = y_max + (y_range * 0.15)
    
    for i, (x, count) in enumerate(zip(x_positions, positions_counts)):
        ax.text(
            x, label_y_position, f'N = {count}',
            ha='center', va='bottom', fontsize=10
        )
    
    # Adjust y-axis limits to accommodate labels without overlapping title
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.35)
    
    # Labels and formatting
    ax.set_xlabel('Pellet Number', fontsize=14)
    ax.set_ylabel('Time Interval Between Pellets [Sec]', fontsize=14)
    ax.set_title(f'Inter-Pellet Interval by Pellet Position - {group_name}', fontsize=16)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(positions_labels)
    ax.set_xlim(-0.5, len(positions_data) - 0.5)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)