"""
This script manages file paths for the FED3 data analysis project and organizes
data sheets into cohorts.
"""
from preprocessing import get_all_sheet_names

fr1_ctrl_path = '../data/FR1_ctrl.xlsx'
fr1_cask_path = '../data/FR1_cask.xlsx'
rev_ctrl_path = '../data/reversal_ctrl.xlsx'
rev_cask_path = '../data/reversal_cask.xlsx'

fr1_male_path = '../data/FR1_male.xlsx'
fr1_female_path = '../data/FR1_female.xlsx'
rev_male_path = '../data/reversal_male.xlsx'
rev_female_path = '../data/reversal_female.xlsx'


fr1_ctrl_raw = get_all_sheet_names(fr1_ctrl_path)
fr1_cask_raw = get_all_sheet_names(fr1_cask_path)
rev_ctrl_raw = get_all_sheet_names(rev_ctrl_path)
rev_cask_raw = get_all_sheet_names(rev_cask_path)

fr1_male_raw = get_all_sheet_names(fr1_male_path)
fr1_female_raw = get_all_sheet_names(fr1_female_path)
rev_male_raw = get_all_sheet_names(rev_male_path)
rev_female_raw = get_all_sheet_names(rev_female_path)

rev_ctrl_raw.remove('C1.M1')
fr1_male_raw.remove('R1M1')
fr1_male_raw.remove('R2M12')

fr1_female_raw.remove('R1M1')
fr1_female_raw.remove('R1M7')
fr1_female_raw.remove('R2M7')

rev_male_raw.remove('R1M1')
rev_male_raw.remove('R1M3')
rev_male_raw.remove('R1M7')

# general loop through
fr1_ctrl_sheets = sorted(fr1_ctrl_raw)
fr1_cask_sheets = sorted(fr1_cask_raw)
rev_ctrl_sheets = sorted(rev_ctrl_raw)
rev_cask_sheets = sorted(rev_cask_raw)

fr1_male_sheets = sorted(fr1_male_raw)
fr1_female_sheets = sorted(fr1_female_raw)
rev_male_sheets = sorted(rev_male_raw)
rev_female_sheets = sorted(rev_female_raw)

# loop by cohorts
# cohort 1: group 1-4; 
# cohort 2: group 5-8; 
# cohort 3: 9-12;
# cohort 4: C groups
def group_by_cohort(sheets):
    """Groups a list of sheet names into cohorts based on their naming convention.

    Args:
        sheets (list): A list of sheet names (e.g., 'C1.M1', 'B2.M3').

    Returns:
        list: A list of lists, where each inner list represents a cohort of sheet names.
    """
    cohorts = [[], [], [], []]
    for sheet_name in sheets:
        group_info = sheet_name.split('.')[0]
        letter_tag, group_idx = group_info[0], int(group_info[1:])
        if letter_tag == 'C':
            cohorts[3].append(sheet_name)
        elif group_idx <= 4:
            cohorts[0].append(sheet_name)
        elif group_idx <= 8:
            cohorts[1].append(sheet_name)
        else:
            cohorts[2].append(sheet_name)

    if [] in cohorts:
        cohorts.remove([])
    return cohorts


fr1_ctrl_cohorts = group_by_cohort(fr1_ctrl_raw)
fr1_cask_cohorts = group_by_cohort(fr1_cask_raw)
rev_ctrl_cohorts = group_by_cohort(rev_ctrl_raw)
rev_cask_cohorts = group_by_cohort(rev_cask_raw)