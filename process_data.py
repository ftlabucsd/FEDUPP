import os
import pandas as pd
import json

# Define the groups and their corresponding data files.
# This structure can be easily modified to include new groups or file types.
GROUPS_CONFIG = {
    'cask': {
        'fr1': 'data/FR1_cask.xlsx',
        'reversal': 'data/reversal_cask.xlsx'
    },
    'ctrl': {
        'fr1': 'data/FR1_ctrl.xlsx',
        'reversal': 'data/reversal_ctrl.xlsx'
    },
    'female': {
        'fr1': 'data/FR1_female.xlsx',
        'reversal': 'data/reversal_female.xlsx'
    },
}

# Define the output directory for the processed data.
OUTPUT_DIR = 'sample_data'
# Define the name of the JSON mapping file.
JSON_MAP_FILE = 'data_map.json'

def process_data():
    """
    Processes and reorganizes mouse behavioral data from Excel files into a structured
    directory of CSV files and creates a JSON file mapping the new organization.
    """
    # Create the main output directory if it doesn't already exist.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mouse_map = {}
    json_output = {
        'groups': {},
        'mice': {}
    }
    mouse_counter = 1

    # Iterate over each group defined in the configuration.
    for group, files in GROUPS_CONFIG.items():
        json_output['groups'][group] = []
        # Iterate over each file type (e.g., 'fr1', 'reversal') for the group.
        for file_type, file_path in files.items():
            # Check if the source file exists before trying to process it.
            if not os.path.exists(file_path):
                print(f"Warning: File not found, skipping: {file_path}")
                continue

            # Load the Excel file.
            xls = pd.ExcelFile(file_path)
            # Process each sheet (each sheet corresponds to a mouse).
            for sheet_name in xls.sheet_names:
                # Create a unique identifier for each mouse based on its group and original ID.
                original_mouse_id = f"{group}_{sheet_name}"

                # If this is the first time seeing this mouse, assign a new ID and create its directory.
                if original_mouse_id not in mouse_map:
                    new_id = f"M{mouse_counter}"
                    mouse_map[original_mouse_id] = new_id
                    mouse_counter += 1
                    
                    # Add the new mouse ID to its corresponding group in the JSON output.
                    json_output['groups'][group].append(new_id)
                    # Initialize the mouse's data in the JSON output.
                    json_output['mice'][new_id] = {
                        'original_id': sheet_name,
                        'group': group,
                        'files': {}
                    }
                    
                    # Create the directory for the new mouse ID.
                    os.makedirs(os.path.join(OUTPUT_DIR, new_id), exist_ok=True)
                
                # Retrieve the new ID for the current mouse.
                new_id = mouse_map[original_mouse_id]
                
                # Read the data from the sheet.
                df = pd.read_excel(xls, sheet_name=sheet_name)
                
                # Define the path for the output CSV file.
                output_csv_path = os.path.join(OUTPUT_DIR, new_id, f"{file_type}.csv")
                # Save the data to the CSV file.
                df.to_csv(output_csv_path, index=False)
                
                # Record the path to the new CSV file in the JSON output.
                json_output['mice'][new_id]['files'][file_type] = output_csv_path

    # Write the mapping data to a JSON file with human-readable formatting.
    with open(JSON_MAP_FILE, 'w') as f:
        json.dump(json_output, f, indent=4)

    print(f"Data processing complete. Decomposed data is in '{OUTPUT_DIR}' directory.")
    print(f"Mapping of new mouse IDs is saved in '{JSON_MAP_FILE}'.")

if __name__ == '__main__':
    process_data()
