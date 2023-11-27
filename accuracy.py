import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter, HourLocator, date2num
from tools import get_bhv_num

def read_excel_by_sheet(sheet, parent='../behavior data integrated/Adjusted FED3 Data.xlsx'):
    df = pd.read_excel(parent, sheet_name=sheet)

    df = df[['MM:DD:YYYY hh:mm:ss', 'Event', 'Active_Poke',
         'Cum_Sum', 'Percent_Correct']].rename(columns={'MM:DD:YYYY hh:mm:ss': 'Time'}).fillna(0)
    
    df = df.replace({'RightWithPellet': 'Right', 'LeftWithPellet': 'Left'})
    df['Percent_Correct'] *= 100
    df['Time'] = pd.to_datetime(df['Time'])
    mask = (df['Percent_Correct'] != 0) & (df['Cum_Sum'] != 0)
    df = df[mask]

    return df

def graph_cumulative_acc(mice, group):
     plt.figure(figsize=(15, 6), dpi=90)

     cnt = 1
     for each in mice:
          sns.lineplot(data=each, x='Time', y='Percent_Correct', label=f'M{cnt}')
          cnt += 1
     plt.grid()
     plt.title(f'Changes in Correction Rate for Control Group {group}', fontsize=24)
     plt.xlabel('Time', fontsize=16)
     plt.ylabel('Correct Rate', fontsize=16)
     plt.yticks(range(0, 110, 10))
     plt.legend()
     legend = plt.legend(title='Mice', fontsize=10)
     legend.get_title().set_fontsize(12)
     plt.show()

def cumulative_pellets_meals(data, bhv, num):
     plt.figure(figsize=(15, 6), dpi=90)

     sns.lineplot(data=data, x='Time', y='Cum_Sum', label='M1')

     plt.grid()
     plt.title(f'Cumulative Sum of Pellet for Control Group {bhv} Mice {num}', fontsize=22)
     plt.xlabel('Time', fontsize=16)
     plt.ylabel('Cumulative Percentage', fontsize=16)
     plt.legend()
     legend = plt.legend(title='Mice', fontsize=10)
     legend.get_title().set_fontsize(12)
     plt.show()

def calculate_accuracy(group):
    total_events = len(group)
    matching_events = group[group['Event'] == group['Active_Poke']]
    matching_count = len(matching_events)
    
    if total_events == 0:
        return 0
    else:
        return (matching_count / total_events) * 100

def instant_acc(sheet, parent='../behavior data integrated/Adjusted FED3 Data.xlsx'):
     df = pd.read_excel(parent, sheet_name=sheet)
     df = df.replace({'RightWithPellet': 'Right', 'LeftWithPellet': 'Left'})
     df = df[['MM:DD:YYYY hh:mm:ss', 'Event', 'Active_Poke', 'Cum_Sum']].rename(
          columns={'MM:DD:YYYY hh:mm:ss':'Time'}).fillna(0)
     
     # remove pellets, over-consumption of pellets
     df = df[df['Event'] != 'Pellet'].reset_index()
     df['Time'] = pd.to_datetime(df['Time'])

     if (df['Time'].loc[1] - df['Time'].loc[0]).total_seconds() / 3600 > 2:
          df = df[1:].reset_index()
     
     idx = 0
     for each in df.itertuples():
          idx += 1
          if each[-1] > 1:
               break
     df = df[:idx]
     
     # Resample the data to hourly intervals and apply the accuracy calculation function
     df.set_index('Time', inplace=True)
     result = df.resample('1H').apply(calculate_accuracy).reset_index().rename(columns={0: 'Accuracy'})
     result['Time'] = pd.to_datetime(result['Time'])

     return result, get_bhv_num(sheet)

def graph_instant_acc(data, bhv, num):
    plt.figure(figsize=(13, 7), dpi=90)

    ax = sns.barplot(x=data['Time'], y=data['Accuracy'], color="skyblue", width=0.6, label='Accuracy')

    xtick_positions = ax.get_xticks()

    hourly_labels = [label.strftime('%H:%M') for label in data['Time'] if label.minute == 0]
    hourly_positions = [pos for pos, label in zip(xtick_positions, data['Time']) if label.minute == 0]
    ax.set_xticks(hourly_positions)  # Set the tick positions to match the hourly intervals
    ax.set_xticklabels(hourly_labels, rotation=45, horizontalalignment='right')  # Set the tick labels to hourly format
    
    # Locate the x-coordinates for the specified times
    dark = []
    temp = {}
    for idx, tick in enumerate(hourly_labels):
        if len(temp) != 0 and tick == '07:00':
            temp['morning'] = hourly_positions[idx]
        elif tick == '19:00':
            temp['evening'] = hourly_positions[idx]
        
        if len(temp) == 2:
            dark.append(temp)
            temp = {}

    # start at one time, but the end did not stop
    if len(temp) == 1:
        # if 'morning' in temp.keys():
        #     temp['evening'] = hourly_positions[0]
        # else:
        temp['morning'] = len(data)-1
        dark.append(temp)

    for idx, each in enumerate(dark):
        stamps = list(each.values())
        if idx == 0:
            ax.axvspan(stamps[0], stamps[1], color='grey', alpha=0.4, label='Night')
        else:
            ax.axvspan(stamps[0], stamps[1], color='grey', alpha=0.4)


    plt.xlabel('Time')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy over Time of Group {bhv} Mice {num}')
    plt.legend()
    plt.show()