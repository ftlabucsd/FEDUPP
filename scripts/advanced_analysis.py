
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
    
    Uses block-based meal detection: meals are detected within each block separately,
    ensuring no cross-block meals and consistent alignment with block boundaries.
    
    Args:
        fr1_group_sessions: Dictionary of group -> list of SessionData (FR1)
        rev_group_sessions: Dictionary of group -> list of SessionData (REV)
        export_root: Directory to save figures
        day_limit (int): Trim sessions to this many days when splitting blocks to
            stay consistent with the rest of the reversal analyses.
    """
    from scripts.preprocessing import SessionData
    from scripts.meals import find_meals_by_blocks
    
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
        # Track meal counts per block across all sessions in this group
        per_block_meal_counts: list[int] = []
        # Track pellet counts per meal separately for matching and mismatching blocks
        meal_pellets_match: list[int] = []
        meal_pellets_mismatch: list[int] = []
        
        for session in rev_sessions:
            mouse_id = session.key.mouse_id
            fr1_poke = get_fr1_active_poke(mouse_id, fr1_sessions)
            
            if not fr1_poke:
                continue  # Skip if no FR1 data
            
            # 1. Limit to requested day window
            raw_data = session.raw.copy()
            if 'Time_passed' in raw_data.columns and day_limit is not None:
                window_mask = raw_data['Time_passed'] < timedelta(days=day_limit)
                raw_data = raw_data[window_mask].copy()
            
            # 2. Split data into blocks
            blocks = split_data_to_blocks(raw_data, day=day_limit)
            
            # Drop the final block to avoid partially observed end-of-session blocks
            if len(blocks) > 1:
                blocks = blocks[:-1]
            
            # 3. Detect meals within each block separately using find_meals_by_blocks
            # This ensures no cross-block meals
            _, _, block_meal_info = find_meals_by_blocks(
                blocks,
                time_threshold=60,
                pellet_threshold=2,
                method='paper',
                accuracy_threshold=-1.0,  # Include all meals regardless of accuracy
            )
            # Record meal counts for summary statistics
            for info in block_meal_info:
                per_block_meal_counts.append(len(info.get('meals', [])))
            
            # 4. Compute block boundaries with a small tail padding to avoid
            # compressing the final meal to ~100% when the active poke switches
            block_boundaries = []
            PAD_SECONDS = 60  # pad tail by one meal window
            for block_idx, block in enumerate(blocks):
                if block.empty:
                    continue
                
                # Block start: first event time in this block
                block_start = block['Time'].iloc[0]
                
                # Base block end: start of next block (transition point) when available
                # otherwise last event time of this block.
                if block_idx < len(blocks) - 1 and not blocks[block_idx + 1].empty:
                    next_start = blocks[block_idx + 1]['Time'].iloc[0]
                    block_end = next_start
                    gap_to_next = (next_start - block['Time'].iloc[-1]).total_seconds()
                    # If the transition is immediate (very small gap), add a short padding
                    # so that the last meal midpoint is not forced to 100%.
                    if gap_to_next < PAD_SECONDS:
                        block_end = block_end + timedelta(seconds=PAD_SECONDS - gap_to_next)
                else:
                    block_end = block['Time'].iloc[-1] + timedelta(seconds=PAD_SECONDS)
                
                block_boundaries.append({
                    'idx': block_idx,
                    'start': block_start,
                    'end': block_end,
                    'duration': (block_end - block_start).total_seconds(),
                    'active_poke': block['Active_Poke'].iloc[0],
                })
            
            # 5. Process each block and its meals
            for boundary in block_boundaries:
                block_idx = boundary['idx']
                block_start = boundary['start']
                block_end = boundary['end']
                block_duration = boundary['duration']
                
                if block_duration <= 0:
                    continue
                
                block_active_poke = boundary['active_poke']
                is_match = (block_active_poke == fr1_poke)
                target_dict = data_match if is_match else data_mismatch
                
                # Get meals for this block from pre-computed block_meal_info
                if block_idx >= len(block_meal_info):
                    continue
                    
                block_meals = block_meal_info[block_idx]['meals']
                block_accs = block_meal_info[block_idx]['meal_acc']
                
                # Get the block data for counting pellets per meal
                block_data = blocks[block_idx]
                pellet_events = block_data[block_data['Event'] == 'Pellet']
                
                # Process each meal in this block
                for (m_start, m_end), acc in zip(block_meals, block_accs):
                    # Count pellets in this meal
                    meal_pellet_mask = (
                        (pellet_events['Time'] >= m_start) &
                        (pellet_events['Time'] <= m_end)
                    )
                    pellet_count = meal_pellet_mask.sum()
                    
                    # Track pellet count for matching vs mismatching blocks
                    if is_match:
                        meal_pellets_match.append(pellet_count)
                    else:
                        meal_pellets_mismatch.append(pellet_count)
                    
                    # Calculate meal midpoint position within block
                    rel_pos = ((m_start - block_start).total_seconds() / block_duration) * 100
                    
                    if acc >= 50.0:
                        target_dict['High'].append(rel_pos)
                    else:
                        target_dict['Low'].append(rel_pos)
        
        # Print separate statistics for pellets per meal in matching vs mismatching blocks
        if meal_pellets_match:
            mean_match = float(np.mean(meal_pellets_match))
            median_match = float(np.median(meal_pellets_match))
            print(
                f"[Group {group}] Matching blocks - pellets per meal: "
                f"mean={mean_match:.2f}, median={median_match:.2f}, "
                f"n_meals={len(meal_pellets_match)}"
            )
        if meal_pellets_mismatch:
            mean_mismatch = float(np.mean(meal_pellets_mismatch))
            median_mismatch = float(np.median(meal_pellets_mismatch))
            print(
                f"[Group {group}] Mismatching blocks - pellets per meal: "
                f"mean={mean_mismatch:.2f}, median={median_mismatch:.2f}, "
                f"n_meals={len(meal_pellets_mismatch)}"
            )
        
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
        
        # Plot meal size (pellets per meal) distribution for matching vs mismatching blocks
        if meal_pellets_match or meal_pellets_mismatch:
            fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
            
            # Determine bin edges based on the combined data range
            # Use -0.5 offset so integer values are centered in each bin
            all_pellet_counts = meal_pellets_match + meal_pellets_mismatch
            if all_pellet_counts:
                max_count = max(all_pellet_counts)
                # Create bins centered on integers: -0.5, 0.5, 1.5, ..., max+0.5
                pellet_bins = np.arange(-0.5, max_count + 1.5, 1)
            else:
                pellet_bins = np.arange(-0.5, 11.5, 1)
            
            # Color scheme matching the main plot style
            match_color = '#3274A1'  # Blue
            mismatch_color = '#E1812C'  # Orange
            
            # Set x-tick positions at integer values (centered on bars)
            if all_pellet_counts:
                tick_positions = np.arange(0, max_count + 1, 1)
            else:
                tick_positions = np.arange(0, 11, 1)
            
            # Plot Matching blocks - pellets per meal distribution
            if meal_pellets_match:
                ax3.hist(meal_pellets_match, bins=pellet_bins, color=match_color,
                         alpha=0.7, edgecolor='black', linewidth=0.8)
                mean_match = np.mean(meal_pellets_match)
                median_match = np.median(meal_pellets_match)
                ax3.axvline(mean_match, color='#d62728', linestyle='--', linewidth=2,
                            label=f'Mean: {mean_match:.2f}')
                ax3.axvline(median_match, color='#2ca02c', linestyle=':', linewidth=2,
                            label=f'Median: {median_match:.1f}')
                ax3.legend(loc='upper right', fontsize=10)
            ax3.set_title(f'Group {group}: Matching FR1 Active Poke\nMeal Size Distribution',
                          fontsize=12, fontweight='bold')
            ax3.set_xlabel('Number of Pellets per Meal', fontsize=11)
            ax3.set_ylabel('Frequency (Number of Meals)', fontsize=11)
            ax3.set_xlim(-0.5, max_count + 0.5 if all_pellet_counts else 10.5)
            ax3.set_xticks(tick_positions)
            ax3.grid(axis='y', alpha=0.3)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            
            # Plot Mismatching blocks - pellets per meal distribution
            if meal_pellets_mismatch:
                ax4.hist(meal_pellets_mismatch, bins=pellet_bins, color=mismatch_color,
                         alpha=0.7, edgecolor='black', linewidth=0.8)
                mean_mismatch = np.mean(meal_pellets_mismatch)
                median_mismatch = np.median(meal_pellets_mismatch)
                ax4.axvline(mean_mismatch, color='#d62728', linestyle='--', linewidth=2,
                            label=f'Mean: {mean_mismatch:.2f}')
                ax4.axvline(median_mismatch, color='#2ca02c', linestyle=':', linewidth=2,
                            label=f'Median: {median_mismatch:.1f}')
                ax4.legend(loc='upper right', fontsize=10)
            ax4.set_title(f'Group {group}: Mismatching FR1 Active Poke\nMeal Size Distribution',
                          fontsize=12, fontweight='bold')
            ax4.set_xlabel('Number of Pellets per Meal', fontsize=11)
            ax4.set_xlim(-0.5, max_count + 0.5 if all_pellet_counts else 10.5)
            ax4.set_xticks(tick_positions)
            ax4.grid(axis='y', alpha=0.3)
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            
            plt.tight_layout()
            if export_root:
                path = f"{export_root}/rev_block_meal_size_dist_{group}.svg"
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
