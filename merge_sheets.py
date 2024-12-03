import os
import pandas as pd

def categorize_and_append_data(root_directory, ctrl_samples):
    # Define subcategory file mappings
    subcategory_files = {
        'FR1_cask': 'FR1_cask.xlsx',
        'FR1_ctrl': 'FR1_ctrl.xlsx',
        'reversal_cask': 'reversal_cask.xlsx',
        'reversal_ctrl': 'reversal_ctrl.xlsx'
    }

    # Process each subfolder in the root directory
    for subfolder in os.listdir(root_directory):
        subfolder_path = os.path.join(root_directory, subfolder)

        # Skip if not a directory
        if not os.path.isdir(subfolder_path):
            continue

        # Determine group (ctrl or cask)
        group = 'ctrl' if subfolder in ctrl_samples else 'cask'

        # Process CSV files in the subfolder
        for csv_file in os.listdir(subfolder_path):
            if csv_file.endswith('.csv') or csv_file.endswith('.CSV'):
                csv_path = os.path.join(subfolder_path, csv_file)

                # Read CSV file
                try:
                    data = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
                    continue

                session_type = 'FR1' if 'FR1' in data['Session_type'].values[0] else 'reversal'
                print(f'Processing {subfolder} in {group} group of {session_type} data')

                subcategory = f"{session_type}_{group}"
                excel_file = subcategory_files[subcategory]
                sheet_name = subfolder[:2] + '.' + subfolder[2:]

                # Append data to a new sheet in the Excel file
                try:
                    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a') as writer:
                        # Load workbook to check existing sheets
                        book = writer.book
                        if sheet_name in book.sheetnames:
                            print(f"Sheet {sheet_name} already exists in {excel_file}. Skipping...")
                            continue
                        # Write to a new sheet
                        data.to_excel(writer, index=False, sheet_name=sheet_name)
                        print(f"Appended data to new sheet {sheet_name} in {excel_file}")
                except Exception as e:
                    print(f"Error writing to {excel_file}: {e}")


root_directory = "./CASK BHV mice cages 5-9"  # Replace with your samples directory
ctrl_samples = ["C5M2", "C6M1", "C6M4", "C7M1", "C8M1", "C8M4", "C9M1", "C9M3"]  # Replace with the list of control sample subfolder names

categorize_and_append_data(root_directory, ctrl_samples)
