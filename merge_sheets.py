import os
import pandas as pd
from pathlib import Path

def categorize_and_append_data(root_directory):
    subcategory_files = {
        'FR1_male': './data/FR1_male.xlsx',
        'FR1_female': './data/FR1_female.xlsx',
        'reversal_male': './data/reversal_male.xlsx',
        'reversal_female': './data/reversal_female.xlsx'
    }

    # create empty files
    for file in subcategory_files.values():
        pd.DataFrame().to_excel(file, index=False)
    
    # Female_1, Male_1, etc
    for cohort in os.listdir(root_directory):
        cohort_path = os.path.join(root_directory, cohort)
        session_idx = cohort[-1]
        
        if not os.path.isdir(cohort_path): continue

        # Determine gender
        group = 'female' if 'Fe' in cohort else 'male'

        # Process mice files - subfolders like C1M1, C2M3, etc
        cnt = 1 # record mouse index
        for subfolder in os.listdir(cohort_path):
            mouse_path = os.path.join(cohort_path, subfolder)
            if not os.path.isdir(mouse_path): continue

            # process each csv file
            for csv_file in os.listdir(mouse_path):
                if csv_file.endswith('.csv') or csv_file.endswith('.CSV'):
                    csv_path = os.path.join(mouse_path, csv_file)
                    try:
                        data = pd.read_csv(csv_path)
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")
                        continue

                    session_type = 'FR1' if 'FR1' in data['Session_type'].values[0] else 'reversal'
                    print(f'Processing {subfolder} in {group} group of {session_type} data')

                    subcategory = f"{session_type}_{group}"
                    excel_file = subcategory_files[subcategory]
                    sheet_name = f'R{session_idx}M{cnt}'

                    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a') as writer:
                        # Load workbook to check existing sheets
                        book = writer.book
                        print(book)
                        if sheet_name in book.sheetnames:
                            print(f"Sheet {sheet_name} already exists in {excel_file}. Skipping...")
                            continue
                        data.to_excel(writer, index=False, sheet_name=sheet_name)
                        print(f"Appended data to new sheet {sheet_name} in {excel_file}")
            cnt += 1
        cnt = 1 # reset index after each cohort



root_directory = "./wild_type_raw"  # Replace with your samples directory
categorize_and_append_data(root_directory)
