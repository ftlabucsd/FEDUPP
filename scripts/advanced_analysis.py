
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import timedelta
import warnings
from scripts.meals import find_meals_paper
from scripts.direction_transition import split_data_to_blocks

def plot_fr1_meal_accuracy_distribution(fr1_group_sessions, bin_size_hours=1, export_path=None):
    """
    Task 1: Plot frequency of meals with <50% and >=50% accuracy over time for FR1 sessions.
    
    Args:
        fr1_group_sessions: Dictionary of group -> list of SessionData
        bin_size_hours: Size of time bins in hours (default 1)
        export_path: Path to save the figure
    """
    groups = list(fr1_group_sessions.keys())
    n_groups = len(groups)
    
    # Prepare figure
    fig, axes = plt.subplots(n_groups, 1, figsize=(12, 4 * n_groups), sharex=True, sharey=True)
    if n_groups == 1:
        axes = [axes]
        
    max_hours = 24 # Standardize to 24 hours
    bins = np.arange(0, max_hours + bin_size_hours, bin_size_hours)
    
    colors = {'High': '#2ca02c', 'Low': '#d62728'} # Green for high, Red for low
    
    for idx, group in enumerate(groups):
        sessions = fr1_group_sessions[group]
        high_acc_times = []
        low_acc_times = []
        
        for session in sessions:
            data = session.raw.copy()
            # Use find_meals_paper with -1.0 threshold to get all valid meals (including 0% acc)
            meals, meal_accs = find_meals_paper(data, accuracy_threshold=-1.0, method='paper')
            
            start_time = data['Time'].iloc[0]
            
            for (m_start, m_end), acc in zip(meals, meal_accs):
                # Calculate hours from start
                time_diff = (m_start - start_time).total_seconds() / 3600
                if time_diff > max_hours:
                    continue
                    
                if acc >= 50.0:
                    high_acc_times.append(time_diff)
                else:
                    low_acc_times.append(time_diff)
        
        ax = axes[idx]
        
        # Plot histograms
        ax.hist([low_acc_times, high_acc_times], bins=bins, stacked=True, 
                color=[colors['Low'], colors['High']], label=['Low Accuracy (<50%)', 'High Accuracy (>=50%)'],
                alpha=0.7, edgecolor='black')
        
        ax.set_title(f'Group: {group} - Meal Accuracy Distribution over Time')
        ax.set_ylabel('Frequency (Total Meals)')
        if idx == n_groups - 1:
            ax.set_xlabel('Time from Session Start (Hours)')
        
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
    plt.tight_layout()
    if export_path:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()

def plot_reversal_block_accuracy_distribution(
    fr1_group_sessions,
    rev_group_sessions,
    export_root=None,
    day_limit: int = 3,
):
    """
    Task 2: Analyze influence of FR1 experience on Reversal block performance.
    Plots distribution of meal accuracies within blocks (Match vs Mismatch FR1 active poke).
    
    Args:
        fr1_group_sessions: Dictionary of group -> list of SessionData (FR1)
        rev_group_sessions: Dictionary of group -> list of SessionData (REV)
        export_root: Directory to save figures
        day_limit (int): Trim sessions to this many days when splitting blocks to
            stay consistent with the rest of the reversal analyses.
    """
    from scripts.preprocessing import SessionData
    
    groups = list(rev_group_sessions.keys())
    
    # Helper to get FR1 active poke for a mouse
    def get_fr1_active_poke(mouse_id, group_fr1_sessions):
        for s in group_fr1_sessions:
            if s.key.mouse_id == mouse_id:
                # active poke is usually constant in FR1, check first row
                return s.raw['Active_Poke'].iloc[0]
        return None

    # Bins for block progress (0-100%, 5% steps)
    bins = np.arange(0, 101, 5)
    
    for group in groups:
        if group not in fr1_group_sessions:
            continue
            
        fr1_sessions = fr1_group_sessions[group]
        rev_sessions = rev_group_sessions[group]
        
        # Store data: separate for Match and Mismatch
        # Structure: List of relative positions (0-100) for High and Low acc meals
        data_match = {'High': [], 'Low': []}
        data_mismatch = {'High': [], 'Low': []}
        
        for session in rev_sessions:
            mouse_id = session.key.mouse_id
            fr1_poke = get_fr1_active_poke(mouse_id, fr1_sessions)
            
            if not fr1_poke:
                continue # Skip if no FR1 data
            
            # 1. Limit to requested day window and detect meals
            raw_data = session.raw.copy()
            if 'Time_passed' in raw_data.columns and day_limit is not None:
                window_mask = raw_data['Time_passed'] < timedelta(days=day_limit)
                raw_data = raw_data[window_mask].copy()
            all_meals, all_meal_accs = find_meals_paper(
                raw_data,
                accuracy_threshold=-1.0,
                method='paper',
            )
            
            if not all_meals:
                continue

            # 2. Get blocks
            blocks = split_data_to_blocks(raw_data, day=day_limit)

            # 3. Pre-calculate block boundaries using start of next block as the effective end
            block_windows = []
            for b_idx, block in enumerate(blocks):
                if block.empty:
                    continue
                window_start = block['Time'].iloc[0]
                window_end = block['Time'].iloc[-1]
                duration = (window_end - window_start).total_seconds()
                if duration <= 0:
                    continue
                block_windows.append({
                    'start': window_start,
                    'end': window_end,
                    'active_poke': block['Active_Poke'].iloc[0],
                    'duration': duration,
                    'index': b_idx + 1,
                })
            
            if not block_windows:
                continue
            
            # 4. Assign each meal to a block
            for (m_start, m_end), acc in zip(all_meals, all_meal_accs):
                meal_duration = (m_end - m_start).total_seconds()
                if meal_duration <= 0:
                    continue
                
                # Prefer assigning by midpoint so each meal is counted once.
                meal_mid = m_start + (m_end - m_start) / 2
                assigned_block = next(
                    (window for window in block_windows if window['start'] <= meal_mid <= window['end']),
                    None,
                )
                
                if assigned_block is None:
                    # Fallback to overlap ratio when midpoint lands in a gap
                    best_block = None
                    best_overlap = 0.0
                    for window in block_windows:
                        overlap_start = max(window['start'], m_start)
                        overlap_end = min(window['end'], m_end)
                        overlap = (overlap_end - overlap_start).total_seconds()
                        if overlap <= 0:
                            continue
                        overlap_ratio = overlap / meal_duration
                        if overlap_ratio > best_overlap:
                            best_overlap = overlap_ratio
                            best_block = window
                    if best_block is None or best_overlap < 0.5:
                        warnings.warn(
                            f"Skipped meal spanning multiple blocks for mouse {mouse_id}; "
                            "unable to find >=50% overlap with any block."
                        )
                        continue
                    assigned_block = best_block
                
                duration = assigned_block['duration']
                if duration <= 0:
                    continue
                
                overlap_start = max(assigned_block['start'], m_start)
                overlap_end = min(assigned_block['end'], m_end)
                if overlap_end <= overlap_start:
                    continue
                overlap_mid = overlap_start + (overlap_end - overlap_start) / 2
                
                # Determine match
                is_match = (assigned_block['active_poke'] == fr1_poke)
                target_dict = data_match if is_match else data_mismatch

                rel_pos = ((overlap_mid - assigned_block['start']).total_seconds() / duration) * 100
                rel_pos = np.clip(rel_pos, 0.0, 100.0)

                if acc >= 50.0:
                    target_dict['High'].append(rel_pos)
                else:
                    target_dict['Low'].append(rel_pos)
        
        # Plotting for this group
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        
        # Plot Match
        ax1.hist([data_match['Low'], data_match['High']], bins=bins, stacked=True,
                 color=['#d62728', '#2ca02c'], label=['Low Acc (<50%)', 'High Acc (>=50%)'],
                 density=False, alpha=0.7, edgecolor='black')
        ax1.set_title(f'Group {group}: Matching FR1 Active Poke')
        ax1.set_xlabel('Block Progress (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_xlim(0, 100)
        ax1.grid(alpha=0.3)
        
        # Plot Mismatch
        ax2.hist([data_mismatch['Low'], data_mismatch['High']], bins=bins, stacked=True,
                 color=['#d62728', '#2ca02c'], label=['Low Acc (<50%)', 'High Acc (>=50%)'],
                 density=False, alpha=0.7, edgecolor='black')
        ax2.set_title(f'Group {group}: Mismatching FR1 Active Poke')
        ax2.set_xlabel('Block Progress (%)')
        ax2.set_xlim(0, 100)
        ax2.grid(alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        if export_root:
            path = f"{export_root}/rev_block_acc_dist_{group}.svg"
            plt.savefig(path, bbox_inches='tight')
        plt.show()
        
def calculate_dispense_delays(
    csv_path,
    *,
    max_retrieval_gap: float = 60.0,
    max_dispense_delay: float = 120.0,
):
    """
    Estimate the mechanical dispense delay for each pellet in a session.
    
    Dispense delay is defined as:
        (Pellet drop time) - (timestamp of the correct active poke that triggered it)
    
    where the drop time is inferred from the pellet retrieval timestamp minus the
    ``Retrieval_Time`` (collect_time) recorded by the FED3 firmware.
    
    Args:
        csv_path: Path to the raw FED3 CSV file.
        max_retrieval_gap: Cap for retrieval latency when ``collect_time`` is
            missing or exceeds the 60s meal definition.
        max_dispense_delay: Ignore poke→drop pairings whose inferred delay exceeds
            this many seconds (typically indicates a mismatched trigger).
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return pd.DataFrame()

    if 'MM:DD:YYYY hh:mm:ss' not in df.columns or 'Event' not in df.columns:
        return pd.DataFrame()
    
    df = df.rename(columns={
        "MM:DD:YYYY hh:mm:ss": "Time",
        "Retrieval_Time": "collect_time",
    })
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df = df.dropna(subset=['Time']).reset_index(drop=True)
    
    # Pre-filter: Mark Timed_out as NaN
    # Note: 'Retrieval_Time' in raw CSV is mixed type (numeric or string "Timed_out")
    if "collect_time" in df.columns:
        df["collect_time"] = pd.to_numeric(df["collect_time"], errors='coerce')
        # 'coerce' turns "Timed_out" into NaN
    else:
        return pd.DataFrame()
    
    df['Event'] = df['Event'].astype(str).str.strip()
    df['Active_Poke'] = df['Active_Poke'].astype(str).str.strip()
    
    dispense_records: list[dict] = []
    last_trigger_time: pd.Timestamp | None = None
    
    for _, row in df.iterrows():
        event = row['Event']
        time_stamp = row['Time']
        
        if event == row['Active_Poke']:
            last_trigger_time = time_stamp
            continue
        
        if event != 'Pellet':
            continue
        
        if last_trigger_time is None:
            continue
        
        collect_time = row['collect_time']
        collect_time = pd.to_numeric(collect_time, errors='coerce')
        capped = False
        if pd.isna(collect_time):
            collect_time = max_retrieval_gap
            capped = True
        elif collect_time < 0:
            continue
        elif collect_time > max_retrieval_gap:
            collect_time = max_retrieval_gap
            capped = True
        
        pellet_time = time_stamp
        drop_time = pellet_time - pd.to_timedelta(float(collect_time), unit='s')
        delay = (drop_time - last_trigger_time).total_seconds()
        
        if delay < 0 or delay > max_dispense_delay:
            last_trigger_time = None
            continue
        
        dispense_records.append({
            'Pellet_Time': pellet_time,
            'Trigger_Time': last_trigger_time,
            'Drop_Time': drop_time,
            'Dispense_Delay': delay,
            'Retrieval_Time': float(collect_time),
            'Retrieval_Was_Capped': capped,
        })
        last_trigger_time = None
    
    if not dispense_records:
        return pd.DataFrame(columns=[
            'Pellet_Time',
            'Trigger_Time',
            'Drop_Time',
            'Dispense_Delay',
            'Retrieval_Time',
            'Retrieval_Was_Capped',
        ])
    return pd.DataFrame(dispense_records)

def plot_meal_dispense_time_correlation(rev_group_sessions, export_root=None):
    """
    Task 3: Correlate Meal Accuracy with Average Dispense Time (collect_time) in Reversal.
    Updated to measure mechanical dispense delays (trigger poke → pellet drop) and
    to cap retrieval latencies according to the 60s meal definition.  Any remaining
    extreme values are logged for debugging instead of silently biasing the plots.
    """
    groups = list(rev_group_sessions.keys())
    MEAL_WINDOW_SECONDS = 60.0
    DISPENSE_ALERT_THRESHOLD = 60.0
    
    for group in groups:
        sessions = rev_group_sessions[group]
        meal_accuracies = []
        avg_dispense_times = []
        
        for session in sessions:
            # Recalculate dispensing delays
            csv_path = session.key.session_path
            dispense_df = calculate_dispense_delays(csv_path)
            
            if dispense_df.empty:
                continue
                
            data = session.raw.copy()
            meals, meal_accs = find_meals_paper(data, accuracy_threshold=-1.0, method='paper')
            
            for (m_start, m_end), acc in zip(meals, meal_accs):
                mask = (
                    (dispense_df['Pellet_Time'] >= m_start) &
                    (dispense_df['Pellet_Time'] <= m_end)
                )
                meal_records = dispense_df.loc[mask].sort_values('Pellet_Time')
                delays = meal_records['Dispense_Delay'].dropna().to_numpy(dtype=float)
                
                if delays.size == 0:
                    continue
                
                retrieval_series = meal_records['Retrieval_Time'].dropna().to_numpy(dtype=float)
                if retrieval_series.size > 1:
                    later_retrieval_sum = retrieval_series[1:].sum()
                    if later_retrieval_sum > MEAL_WINDOW_SECONDS + 1e-6:
                        print(
                            f"[{session.key.session_id}] Sum of retrieval times for pellets 2+ "
                            f"({later_retrieval_sum:.1f}s) exceeded {MEAL_WINDOW_SECONDS}s "
                            f"in meal {m_start}–{m_end}."
                        )
                
                avg_delay = float(np.mean(delays))
                if avg_delay > DISPENSE_ALERT_THRESHOLD:
                    print(
                        f"[{session.key.session_id}] Average dispense delay {avg_delay:.1f}s "
                        f"({len(delays)} pellets) for meal {m_start}–{m_end}."
                    )
                
                if not np.isnan(acc):
                    meal_accuracies.append(acc)
                    avg_dispense_times.append(avg_delay)
        
        if not meal_accuracies:
            print(f"No valid meal data for group {group}")
            continue
            
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=avg_dispense_times, y=meal_accuracies, alpha=0.5, ax=ax, color='blue')
        
        # Regression line
        if len(meal_accuracies) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(avg_dispense_times, meal_accuracies)
            x_vals = np.array(ax.get_xlim())
            y_vals = intercept + slope * x_vals
            ax.plot(x_vals, y_vals, color='red', linestyle='--', 
                    label=f'R={r_value:.3f}, p={p_value:.3e}')
            ax.legend()
            
        ax.set_title(f'Group {group}: Meal Accuracy vs Dispensing Delay')
        ax.set_xlabel('Average Dispensing Delay (s)')
        ax.set_ylabel('Meal Accuracy (%)')
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
        
        if export_root:
            path = f"{export_root}/rev_acc_vs_dispense_delay_{group}.svg"
            plt.savefig(path, bbox_inches='tight')
        plt.show()
