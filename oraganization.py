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
    
    

# root_folder = '/home/ftlab/Desktop/For_Andy/behavior data integrated/CD1 IVSA'
# organize_files(root_folder)


file1 = '/home/ftlab/Desktop/For_Andy/behavior data integrated/mPFC/Fentanyl Tx/Mouse_3/Contingency_Flip/FED000_042924_01.CSV'
file2 = '/home/ftlab/Desktop/For_Andy/behavior data integrated/mPFC/Fentanyl Tx/Mouse_3/Contingency_Flip/FED000_043024_00.CSV'
file3 = '/home/ftlab/Desktop/For_Andy/behavior data integrated/mPFC/Vehicle Tx/Mouse_6/Contingency_Flip/FED000_050124_01.CSV'
output = '/home/ftlab/Desktop/For_Andy/behavior data integrated/mPFC/Fentanyl Tx/Reversal/M3.CSV'
concatenate_csv_files([file1, file2], output)