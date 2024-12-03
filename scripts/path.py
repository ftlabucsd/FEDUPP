from preprocessing import get_all_sheet_names

fr1_ctrl_path = '../data/FR1_ctrl.xlsx'
fr1_cask_path = '../data/FR1_cask.xlsx'
rev_ctrl_path = '../data/reversal_ctrl.xlsx'
rev_cask_path = '../data/reversal_cask.xlsx'

fr1_ctrl_sheets = sorted(get_all_sheet_names(fr1_ctrl_path))
fr1_cask_sheets = sorted(get_all_sheet_names(fr1_cask_path))
rev_ctrl_sheets = sorted(get_all_sheet_names(rev_ctrl_path))
rev_cask_sheets = sorted(get_all_sheet_names(rev_cask_path))
