import os
import openpyxl
import pandas as pd

def update_excel_with_csv(csv_path, excel_path, sheet_name):
    csv_df = pd.read_csv(csv_path)
    # exist then append mode, or write mode
    mode = "a" if os.path.exists(excel_path) else "w"

    with pd.ExcelWriter(excel_path, mode=mode, engine="openpyxl") as writer:
        csv_df.to_excel(writer, sheet_name=sheet_name, index=False)


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


def check_data_by_date(path:str, sheet:str):
    df = pd.read_excel(path, sheet)
    df['MM:DD:YYYY hh:mm:ss'] = pd.to_datetime(df['MM:DD:YYYY hh:mm:ss'])

    index_drop = (df['MM:DD:YYYY hh:mm:ss'].diff() < pd.Timedelta(0)).idxmax()
    df_part1 = df.iloc[:index_drop]
    df_part2 = df.iloc[index_drop:]

    df_corrected = pd.concat([df_part2, df_part1]).reset_index(drop=True)
    df_corrected.to_csv(sheet+'.csv', index=False)


def merge_csvs_to_excel(excel:str, csv_root:str):
    files = os.listdir(csv_root)
    if '.DS_Store' in files: 
        files.remove('.DS_Store')

    # construct group meta
    for file in files:
        path = os.path.join(csv_root, file)
        print(path)
        name = file.split('.')[0]
        sheet = name[:len(name)-2]+'.'+name[-2:]
        update_excel_with_csv(csv_path=path, excel_path=excel, sheet_name=sheet)


def split_ctrl_cask(excel_path:str):
    xls = pd.ExcelFile(excel_path)  
    all_sheets = xls.sheet_names
    ctrl_sheets = dict()
    cask_sheets = dict()
    for sheet in all_sheets:
        if sheet.startswith('B'): 
            group = int(sheet.split('.')[0][1:])
            if group in [2, 3, 5, 7, 11, 12]:
                ctrl_sheets[sheet] = xls.parse(sheet)
            else:
                cask_sheets[sheet] = xls.parse(sheet)
        elif sheet.startswith('C'):
            if sheet in ['C1.M1', 'C1.M2', 'C2.M1', 'C2.M3', 'C3.M4']:
                ctrl_sheets[sheet] = xls.parse(sheet)
            else:
                cask_sheets[sheet] = xls.parse(sheet)

    with pd.ExcelWriter("../FR1_ctrl.xlsx", engine="openpyxl") as writer:
        for sheet_name, data in ctrl_sheets.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    with pd.ExcelWriter("../FR1_cask.xlsx", engine="openpyxl") as writer:
        for sheet_name, data in cask_sheets.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)


def copy_sheet(input_file, target_file, sheet_name):
    try:
        input_wb = openpyxl.load_workbook(input_file, data_only=True)
        if sheet_name not in input_wb.sheetnames:
            print(f"Sheet '{sheet_name}' does not exist in {input_file}")
            return

        input_sheet = input_wb[sheet_name]
        target_wb = openpyxl.load_workbook(target_file)

        if sheet_name in target_wb.sheetnames:
            print(f"Sheet '{sheet_name}' already exists in {target_file}.")
            return

        target_sheet = target_wb.create_sheet(title=sheet_name.strip())
        for row in input_sheet.iter_rows():
            for cell in row:
                target_sheet.cell(row=cell.row, column=cell.column, value=cell.value)

        target_wb.save(target_file)
        print(f"Copied sheet '{sheet_name}' from {input_file} to {target_file}.")
    except Exception as e:
        print(f"Error copying sheet: {e}")


if __name__ == '__main__':
    # merge_csvs_to_excel(excel='../reversal_cask.xlsx', csv_root='../CASK_Data/reversal/cask')
    # split_ctrl_cask('../FR1_collection.xlsx')
    # check_data_by_date('../reversal_ctrl.xlsx', 'C1.M1')
    # copy_sheet('../reversal_cask.xlsx', '../reversal_ctrl.xlsx', 'C5.M2')
    file_1 = '../wild_type_raw/Female_2/M4/FED000_121924_01.CSV'
    file_2 = '../wild_type_raw/Female_2/M4/FED000_121924_02.CSV'
    concatenate_csv_files([file_1, file_2], '../wild_type_raw/Female_2/M4/FED000_121924_03.CSV')
    prep_pellet_count('../wild_type_raw/Female_2/M4/FED000_121924_03.CSV')