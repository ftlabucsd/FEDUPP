import os
import shutil
import pandas as pd

def organize_files(root_folder):
    fr1_folder = os.path.join(root_folder, 'FR1')
    reversal_folder = os.path.join(root_folder, 'Reversal')
    
    # Create the folders if they don't exist
    if not os.path.exists(fr1_folder):
        os.makedirs(fr1_folder)
    if not os.path.exists(reversal_folder):
        os.makedirs(reversal_folder)

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(subdir, file)
            
            if file == 'FEDmode.csv':
                os.remove(file_path)
                print(f'Deleted {file_path}')
            else:
                try:
                    df = pd.read_csv(file_path)
                    if 'Session_type' in df.columns:
                        session_type = df['Session_type'].iloc[0]
                        if 'FR1' in session_type:
                            target_folder = fr1_folder
                        elif 'Rev' in session_type:
                            target_folder = reversal_folder
                        else:
                            continue

                        # Move the file to the appropriate folder
                        target_path = os.path.join(target_folder, file)
                        shutil.move(file_path, target_path)
                        print(f'Moved {file_path} to {target_path}')
                except Exception as e:
                    print(f'Could not process {file_path}: {e}')


def concatenate_csv_files(files:list, output_file:str):
    # Read the two CSV files into DataFrames
    dfs = []
    for file in files:
        dfs.append(pd.read_csv(file))
    
    # Concatenate the DataFrames
    concatenated_df = pd.concat(dfs, ignore_index=True)
    
    # Save the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file, index=False)
    print(f'Concatenated file saved as {output_file}')
    
    
def fr1_rev_split(directory):
    for root, _, files in os.walk(directory):
        # Remove small files
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.getsize(filepath) < 7800:  # File size less than 8kb
                os.remove(filepath)
                print(f"Deleted {filepath} due to insufficient size.")

        # Now process remaining CSV files
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.exists(filepath) and (file.endswith('csv') or file.endswith('CSV')): 
                df = pd.read_csv(filepath)
                session_type = df['Session_type'].iloc[0]
                if 'FR1' in session_type:
                    new_folder = os.path.join(root, 'FR1')
                elif 'Rev' in session_type:
                    new_folder = os.path.join(root, 'Reversal')
                else:
                    continue

                # Create folder if it does not exist
                new_path = os.path.join(new_folder, file)
                os.makedirs(new_folder, exist_ok=True)
                # Move the file
                shutil.move(filepath, new_path)
                print(f"Moved {filepath} to {new_path}")
 

def prep_pellet_count(path: str):
    """when combining two csv files, making the pellet count column increasing
    instead of go to 0 for the new file.

    Args:
        path (str): path of combined csv files
    """
    df = pd.read_csv(path) 
    base_pellet = 0
    base_left = 0
    base_right = 0
    reach_base = False
    prev_pellet = -1
    prev_left = -1
    prev_right = -1


    for idx, row in df.iterrows():
        if reach_base or prev_pellet > row['Pellet_Count']:
            if not reach_base:
                reach_base = True
                base_pellet = prev_pellet
                base_left = prev_left
                base_right = prev_right
            df.at[idx, 'Pellet_Count'] = row['Pellet_Count'] + base_pellet
            df.at[idx, 'Left_Poke_Count'] = row['Left_Poke_Count'] + base_left
            df.at[idx, 'Right_Poke_Count'] = row['Right_Poke_Count'] + base_right

        else:
            prev_pellet = row['Pellet_Count']
            prev_right = row['Right_Poke_Count']
            prev_left = row['Left_Poke_Count']
    print(base_left, base_right ,base_pellet)
    df.to_csv(path[:-4]+'_1.CSV', index=False)
    # return df

    
    

# root_folder = '/home/ftlab/Desktop/For_Andy/behavior data integrated/CD1 IVSA'
# organize_files(root_folder)


file1 = '/Users/yaomingyang/Desktop/FED3-data/behavior data integrated/mPFC/Fentanyl Tx/FR1/FED000_042224_13.CSV'
file2 = '/Users/yaomingyang/Desktop/FED3-data/behavior data integrated/mPFC/Fentanyl Tx/FR1/FED000_042824_00.CSV'
file3 = '/home/ftlab/Desktop/For_Andy/behavior data integrated/mPFC/Vehicle Tx/Mouse_6/Contingency_Flip/FED000_050124_01.CSV'

root = '/home/ftlab/Desktop/For_Andy/mPFC/Fentanyl Tx/Mouse_10/FR1/'
output = '/Users/yaomingyang/Desktop/FED3-data/behavior data integrated/mPFC/Fentanyl Tx/FR1/M2.CSV'
files = [file1, file2]
concatenate_csv_files(files, output)
# prep_pellet_count(output)
prep_pellet_count('/Users/yaomingyang/Desktop/FED3-data/behavior data integrated/mPFC/Fentanyl Tx/FR1/M2.CSV')
# fr1_rev_split('/home/ftlab/Desktop/For_Andy/behavior data integrated/CD1 IVSA')