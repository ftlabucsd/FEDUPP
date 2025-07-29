"""
This script provides functions for analyzing and visualizing accuracy data from
FED3 experiments. It includes methods for calculating cumulative accuracy,
plotting accuracy over time, and comparing accuracy statistics between different
groups of subjects.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import read_excel_by_sheet
import numpy as np
from datetime import datetime, timedelta
import matplotlib.patches as mpatches

def read_and_record(path:str, sheet:str, ending_corr:list, learned_time:list, acc_dict:dict):
    """
    Reads data from a sheet, calculates accuracy at a specific time point,
    and records metrics.

    Args:
        path (str): Path to the Excel file.
        sheet (str): Name of the sheet to read.
        ending_corr (list): List to append the final accuracy to.
        learned_time (list): List to append the learned time to.
        acc_dict (dict): Dictionary to store accuracy by sheet.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    df = read_excel_by_sheet(sheet, path, collect_time=True, 
                             cumulative_accuracy=True, remove_trivial=True, 
                             convert_large=True)
    target_time = timedelta(hours=24, minutes=30)
    df['time_diff'] = (df['Time_passed'] - target_time).abs()
    closest_accuracy = df.loc[df['time_diff'].idxmin(), 'Percent_Correct']
    df = df[:df['time_diff'].idxmin()]
    df.drop(columns=['time_diff'], axis='columns')

    ending_corr.append(closest_accuracy)
    learned_time.append(find_first_learned_time(df))
    acc_dict[sheet] = closest_accuracy
    return df


def plot_cumulative_accuracy(
    groups: list,
    group_labels: list = None,
    bin_size_sec: int = 10,
    export_path: str = None
):
    """
    Plots the mean ± SEM of cumulative accuracy for one or more groups.

    Args:
        groups (list): A list of cohorts, where each cohort is a list of DataFrames.
        group_labels (list, optional): Names for each cohort. Defaults to None.
        bin_size_sec (int, optional): The bin width in seconds for resampling. Defaults to 10.
        export_path (str, optional): The file path to save the figure. Defaults to None.
    """
    if group_labels is None:
        group_labels = [f"G{idx+1}" for idx in range(len(groups))]

    # colors for up to two groups; you can extend this list for more
    palette = ['#425df5', '#f55442', '#42f58c', '#f5e142']

    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)

    for grp_idx, (group, label) in enumerate(zip(groups, group_labels)):
        binned_series = []

        for subj_idx, df in enumerate(group):
            
            ts = df.set_index('Time_passed')['Percent_Correct']

            # 1) resample onto a regular grid of bin_size_sec
            resampled = (
                ts
                .resample(f"{bin_size_sec}S")
                .mean()   # average within each bin
                .ffill()  # forward-fill so it's truly cumulative
            )

            # 2) convert index from Timestamp to hours
            hours = resampled.index.view('int64') / 1e9 / 3600
            resampled.index = hours

            # name it uniquely so concat keeps columns separate
            resampled.name = f"{label}_{subj_idx}"
            binned_series.append(resampled)

        # 3) build wide DataFrame: rows=bins, cols=subjects
        all_binned = pd.concat(binned_series, axis=1)

        # 4) compute count, mean, std, sem
        cutoff = 24 * 3600 // bin_size_sec
        counts  = all_binned.count(axis=1).iloc[:cutoff]
        mean_pc = all_binned.mean(axis=1).iloc[:cutoff]
        std_pc  = all_binned.std(axis=1, ddof=0).iloc[:cutoff]
        sem_pc  = std_pc / np.sqrt(counts)

        # optionally mask SEM where too few subjects
        sem_pc[counts < 2] = np.nan

        # 5) plot
        color = palette[grp_idx % len(palette)]
        ax.plot(
            mean_pc.index,
            mean_pc.values,
            label=f"{label} (n={len(group)})",
            color=color,
            linewidth=2
        )
        ax.fill_between(
            mean_pc.index,
            mean_pc.values - sem_pc.values,
            mean_pc.values + sem_pc.values,
            alpha=0.3,
            color=color
        )

    # cosmetics
    ax.set_xlabel("Time Passed (hours)", fontsize=24)
    ax.set_ylabel("Percent Correct",    fontsize=24)
    ax.set_ylim(30, 100)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.grid(True, linestyle='--', alpha=0.4)

    title = "Cumulative Accuracy"
    if len(group_labels) == 1:
        title += f" for Cohort {group_labels[0]}"
    else:
        title += " for " + " vs. ".join(group_labels)
    ax.set_title(title, fontsize=32)

    ax.legend(fontsize=16, loc='upper left')

    plt.tight_layout()
    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()


def cumulative_pellets_meals(data: pd.DataFrame, bhv: int, num: int):
    """
    Graphs the cumulative pellet counts for a specific mouse.

    Args:
        data (pd.DataFrame): The input DataFrame for a single mouse.
        bhv (int): The group number of the mouse.
        num (int): The index of the mouse.
    """
    plt.figure(figsize=(15, 6), dpi=90)

    sns.lineplot(data=data, x='Time_passed', y='Cum_Sum', label='M1')

    plt.grid()
    plt.title(f'Cumulative Sum of Pellet for Control Group {bhv} Mice {num}', fontsize=22)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Cumulative Percentage (%)', fontsize=16)
    plt.legend()
    legend = plt.legend(title='Mice', fontsize=10)
    legend.get_title().set_fontsize(12)
    plt.show()


def calculate_accuracy(group: pd.DataFrame):
    """
    Calculates the percentage of correct pokes in a given interval.

    Args:
        group (pd.DataFrame): A DataFrame for a specific interval, containing
                              'Event' and 'Active_Poke' columns.

    Returns:
        float: The accuracy percentage (0-100). Returns 0 if there are no events.
    """
    if 'Pellet' in group['Event'].values:
        group = group[group['Event'] != 'Pellet']
        
    total_events = len(group)
    matching_events = group[group['Event'] == group['Active_Poke']]
    matching_count = len(matching_events)
    
    if total_events == 0:
        return 0
    else:
        return (matching_count / total_events) * 100


def find_night_index(hourly_labels:list, rev:bool):
    """
    Finds pairs of indices corresponding to night (7 pm to 7 am) or day intervals.

    Args:
        hourly_labels (list): A list of times in 'H:M' format.
        rev (bool): If True, returns day intervals; otherwise, returns night intervals.

    Returns:
        list: A list of [start, end] index pairs for the specified intervals.
    """
    intervals = []
    in_interval = False
    interval_start_index = None

    for i, time_str in enumerate(hourly_labels):
        current_time = datetime.strptime(time_str, '%H:%M')

        # Determine if the current time is within the night interval (7pm to 7am)
        if current_time.hour >= 19 or current_time.hour < 7:
            if not in_interval:
                # We are starting a new interval
                in_interval = True
                interval_start_index = i
        else:
            if in_interval:
                # We are ending the current interval
                in_interval = False
                intervals.append([interval_start_index, i - 1])
    
    # If the last time in the list is within the interval, close it
    if in_interval:
        intervals.append([interval_start_index, len(hourly_labels) - 1])
    
    hourly_labels = [datetime.strptime(time_str, '%H:%M').hour for time_str in hourly_labels]
    
    if rev:
        result = []
        start = 0
        for interval in intervals:
            if interval[0] > start:
                result.append([start, interval[0] - 1])
            start = interval[1] + 1
        if start <= len(hourly_labels) - 1:
            result.append([start, len(hourly_labels) - 1])
        return result
    return intervals


def find_first_learned_time(data:pd.DataFrame, window_hours=2, accuracy_threshold=0.8): 
    """
    Finds the first time a sustained accuracy threshold is met over a sliding window.

    Args:
        data (pd.DataFrame): The input DataFrame with 'Event', 'Active_Poke', and 'Time_passed'.
        window_hours (int, optional): The duration of the sliding window in hours. Defaults to 2.
        accuracy_threshold (float, optional): The accuracy threshold to be met. Defaults to 0.8.

    Returns:
        float: The time in hours when the learning criterion is first met. Returns the total
               session time if the criterion is never met.
    """
    data = data[data['Event'] != 'Pellet'].reset_index(drop=True)
    data['is_match'] = (data['Event'] == data['Active_Poke']).astype(int)
    data['cumulative_total'] = data['is_match'].expanding().count()
    data['cumulative_match'] = data['is_match'].cumsum()
    window_timedelta = timedelta(hours=window_hours)

    for start_idx in range(len(data)):
        start_time = data.loc[start_idx, 'Time_passed']
        end_time = start_time + window_timedelta

        window_data = data[(data['Time_passed'] >= start_time) &
                           (data['Time_passed'] < end_time) &
                           (data['Event'] != 'Pellet')]
        total_events = len(window_data)
        matching_events = window_data['is_match'].sum()
        
        if total_events > 0:
            accuracy = matching_events / total_events
            if accuracy > accuracy_threshold:
                return start_time.total_seconds() / 3600  # Return the precise time that meets the condition
    # not reach high accuracy -> return session time
    return data.loc[len(data)-1, 'Time_passed'].total_seconds() / 3600


def graph_group_stats(ctrl: list,
                      exp: list,
                      stats_name: str,
                      unit: str,
                      violin_width=0.25,
                      dpi=150,
                      group_names=None,
                      verbose=True,
                      export_path=None):
    """
    Graphs statistics for two groups using violin plots with inset boxplots and scatter plots.

    Args:
        ctrl (list): Data for the control group.
        exp (list): Data for the experimental group.
        stats_name (str): The name of the statistic being plotted (e.g., "Accuracy").
        unit (str): The unit of the statistic (e.g., "%").
        violin_width (float, optional): The width of the violin plots. Defaults to 0.25.
        dpi (int, optional): The resolution of the figure. Defaults to 150.
        group_names (list, optional): Names for the groups. Defaults to ['Control', 'Experiment'].
        verbose (bool, optional): If True, prints summary statistics. Defaults to True.
        export_path (str, optional): The file path to save the figure. Defaults to None.
    """
    # Default group names
    if group_names is None or len(group_names) < 2:
        group_names = ['Control', 'Experiment']
    ctrl_name, exp_name = group_names

    # Summary statistics
    if verbose:
        ctrl_avg = np.mean(ctrl)
        exp_avg  = np.mean(exp)
        ctrl_se  = np.std(ctrl) / np.sqrt(len(ctrl))
        exp_se   = np.std(exp) / np.sqrt(len(exp))
        print(f'{ctrl_name} Size: {len(ctrl)}')
        print(f'{exp_name} Size: {len(exp)}')
        print(f'{ctrl_name} Average: {ctrl_avg:.3f}')
        print(f'{exp_name} Average: {exp_avg:.3f}')
        print(f'{ctrl_name} SE: {ctrl_se:.3f}')
        print(f'{exp_name} SE: {exp_se:.3f}')

    # Figure setup
    fig, ax = plt.subplots(dpi=dpi)
    fig.set_size_inches(6, 6)

    # X positions for the two groups
    x_positions = [0.5, 1.0]
    data = [ctrl, exp]

    # 1) Violin plots (no interior lines/extrema)
    parts = ax.violinplot(data,
                          positions=x_positions,
                          widths=violin_width,
                          showmeans=False,
                          showmedians=False,
                          showextrema=False)
    for i, violin in enumerate(parts['bodies']):
        color = '#38bcf5' if i == 0 else '#38f5a6'
        violin.set_facecolor(color)
        violin.set_edgecolor('black')
        violin.set_alpha(0.6)

    # 2) Boxplots inside the violins
    ax.boxplot(data,
               positions=x_positions,
               widths=violin_width * 0.5,
               showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor='white', edgecolor='black'),
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'))

    # 3) Scatter individual points (jittered)
    jitter_strength = violin_width / 8
    scatter_colors = ['#18c1f5', '#36d15a']
    for x, group_data, c in zip(x_positions, data, scatter_colors):
        x_j = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(group_data))
        ax.scatter(x_j, group_data, marker='o', zorder=3, color=c, alpha=0.8)

    # Legend
    ctrl_patch = mpatches.Patch(color='#38bcf5', alpha=0.6, label=f'{ctrl_name} (n={len(ctrl)})')
    exp_patch  = mpatches.Patch(color='#38f5a6', alpha=0.6, label=f'{exp_name} (n={len(exp)})')
    ax.legend(handles=[ctrl_patch, exp_patch])

    # Labels & title
    ax.set_xlabel('Groups', fontsize=14)
    ax.set_ylabel(f'{stats_name} ({unit})', fontsize=14)
    ax.set_title(f'{stats_name} Distribution: {ctrl_name} vs. {exp_name}', fontsize=20)

    # X‐axis ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_names)
    ax.set_xlim(0, 1.5)

    # Save or display
    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()


def graph_single_stats(data: list, stats_name: str, unit: str, 
                       violin_width=0.25, dpi=150, group_name=None, verbose=True, export_path=None):
    """
    Graphs statistics for a single group using a violin plot with an inset boxplot and scatter plot.

    Args:
        data (list): The data for the group.
        stats_name (str): The name of the statistic being plotted.
        unit (str): The unit of the statistic.
        violin_width (float, optional): The width of the violin plot. Defaults to 0.25.
        dpi (int, optional): The resolution of the figure. Defaults to 150.
        group_name (str, optional): The name of the group. Defaults to None.
        verbose (bool, optional): If True, prints summary statistics. Defaults to True.
        export_path (str, optional): The file path to save the figure. Defaults to None.
    """
    # Summary stats
    average = np.mean(data)
    std_error = np.std(data) / np.sqrt(len(data))

    if group_name is None:
        group_name = 'Group'
    if verbose:
        print(f'{group_name} Size: {len(data)}')
        print(f'{group_name} Average: {average:.3f}')
        print(f'{group_name} SE: {std_error:.3f}')
    
    # Set up figure
    fig, ax = plt.subplots(dpi=dpi)
    fig.set_size_inches(6, 6)
    x = 0.5  # single group position

    # 1) Violin (no interior lines)
    parts = ax.violinplot([data],
                          positions=[x],
                          widths=violin_width,
                          showmeans=False,
                          showmedians=False,
                          showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#38bcf5')
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)

    # 2) Boxplot inside
    ax.boxplot(data,
               positions=[x],
               widths=violin_width * 0.5,
               showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor='white', edgecolor='black'),
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'))

    # 3) Scatter the individual points (jittered)
    jitter_strength = violin_width / 8
    x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(data))
    ax.scatter(x_jittered, data, marker='o', zorder=3, color='blue', alpha=0.8)

    # Legend
    patch = mpatches.Patch(color='#38bcf5', alpha=0.6, label=f'{group_name} (n = {len(data)})')
    ax.legend(handles=[patch])

    # Labels & formatting
    ax.set_xlabel('Groups', fontsize=14)
    ax.set_ylabel(f'{stats_name} ({unit})', fontsize=14)
    ax.set_title(f'{stats_name} Distribution for {group_name}', fontsize=20)
    ax.set_xticks([x])
    ax.set_xticklabels([group_name])
    ax.set_xlim(0, 1)

    # Save or show
    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()