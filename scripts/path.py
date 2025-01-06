from preprocessing import get_all_sheet_names

fr1_ctrl_path = '../data/FR1_ctrl.xlsx'
fr1_cask_path = '../data/FR1_cask.xlsx'
rev_ctrl_path = '../data/reversal_ctrl.xlsx'
rev_cask_path = '../data/reversal_cask.xlsx'

fr1_ctrl_raw = get_all_sheet_names(fr1_ctrl_path)
fr1_cask_raw = get_all_sheet_names(fr1_cask_path)
rev_ctrl_raw = get_all_sheet_names(rev_ctrl_path)
rev_cask_raw = get_all_sheet_names(rev_cask_path)

# general loop through
fr1_ctrl_sheets = sorted(fr1_ctrl_raw)
fr1_cask_sheets = sorted(fr1_cask_raw)
rev_ctrl_sheets = sorted(rev_ctrl_raw)
rev_cask_sheets = sorted(rev_cask_raw)

# loop by cohorts
# cohort 1: group 1-4; 
# cohort 2: group 5-8; 
# cohort 3: 9-12;
# cohort 4: C groups
def group_by_cohort(sheets):
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
# print(fr1_ctrl_cohorts,'\n')
# print(fr1_cask_cohorts,'\n')
# print(rev_ctrl_cohorts,'\n')
# print(rev_cask_cohorts,'\n')