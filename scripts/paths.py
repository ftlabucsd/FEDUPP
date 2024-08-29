import os


def list_files(root:str, direct_access=True):
    file_paths = []
    
    if direct_access:
        temp = os.listdir(root)
        files = []
        try:
            for item in temp:
                if not item.startswith('.'):
                    files.append(item)
        except:
            pass      
        file_paths = [os.path.join(root, file) for file in files]
    else:
        for root, _, files in os.walk(root):
            try:
                for item in files:
                    if item.startswith('.'):
                        files.remove(item)            
            except:
                pass
            
            for file in files:
                file_paths.append(os.path.join(root, file))
    
    empty = ''
    if empty in file_paths:
        file_paths.remove(empty)

    return sorted(file_paths)


def find_condition(root:str, fr1:bool):
    condition = 'FR1' if fr1 else 'Reversal'
    dirs = os.listdir(root)
    files = []
    for dir in dirs:
        dir = os.path.join(root, dir)
        if os.path.isdir(dir):
            condition_path = os.path.join(root, dir, condition)
            subs = os.listdir(condition_path)
            for sub in subs:
                if not sub.startswith('.'):
                    files.append(os.path.join(condition_path, sub))

    return files 


def remove_ivsa_bad_data(files:list, control:bool, fr1:bool):
    if fr1:
        if control:
            bad = ['43', '52', '45']
        else:
            bad = ['46', '49']
    else:
        if control:
            bad = ['43', '52']
        else:
            return files
    
    ans = []
    for file in files:
        index = file.split('/')[-3][:2]
        if index not in bad:
            ans.append(file)
    return ans
    

fr1_ctrl_sheet = [
    'B3.M1','B3.M2','B3.M3','B3.M4',
    'B5.M1','B5.M2','B5.M3','B5.M4',
    'B7.M2','B7.M3','B7.M4'   
]

fr1_cask_sheet = [
    'B4.M1', 'B4.M2', 'B4.M3', 'B4.M4',
    'B6.M1', 'B6.M2', 'B6.M3', 'B6.M4',
    'B8.M1', 'B8.M2', 'B8.M3'
]

root = '/home/ftlab/Desktop/For_Andy/FED3-data'
# root = '/Users/yaomingyang/Desktop/FED3-data'

contigency_flip_ctrl = list_files(root=f'{root}/behavior data integrated/CASK/reversal/ctrl', direct_access=False)
contigency_flip_cask = list_files(root=f'{root}/behavior data integrated/CASK/reversal/cask', direct_access=False)


fr1_cask_csvs = list_files(root=f'{root}/behavior data integrated/CASK/FR1/cask', direct_access=True)
fr1_ctrl_csvs = list_files(root=f'{root}/behavior data integrated/CASK/FR1/ctrl', direct_access=True)

fr1_ivsa = find_condition(root=f'{root}/behavior data integrated/CD1 IVSA/', fr1=True)
reversal_ivsa = find_condition(root=f'{root}/behavior data integrated/CD1 IVSA/', fr1=False)

fr1_fent = list_files(root=f'{root}/behavior data integrated/mPFC/Fentanyl Tx/FR1', direct_access=True)
reversal_fent = list_files(root=f'{root}/behavior data integrated/mPFC/Fentanyl Tx/Reversal', direct_access=True)

fr1_veh = list_files(root=f'{root}/behavior data integrated/mPFC/Vehicle Tx/FR1', direct_access=True)
reversal_veh = list_files(root=f'{root}/behavior data integrated/mPFC/Vehicle Tx/Reversal', direct_access=True)

fr1_ivsa_ctrl = []
fr1_ivsa_exp = []
reversal_ivsa_ctrl = []
reversal_ivsa_exp = []

for i in fr1_ivsa:
    if i.split('/')[-3][:2] in ['41', '43', '44', '45', '50', '51', '52', '53']:
        fr1_ivsa_ctrl.append(i)
    else:
        fr1_ivsa_exp.append(i)

for i in reversal_ivsa:
    if i.split('/')[-3][:2] in ['41', '43', '44', '45', '50', '51', '52', '53']:
        reversal_ivsa_ctrl.append(i)
    else:
        reversal_ivsa_exp.append(i)

fr1_ivsa_ctrl = remove_ivsa_bad_data(fr1_ivsa_ctrl, True, True)
fr1_ivsa_exp = remove_ivsa_bad_data(fr1_ivsa_exp, False, True)
reversal_ivsa_ctrl = remove_ivsa_bad_data(reversal_ivsa_ctrl, True, False)
reversal_ivsa_exp = remove_ivsa_bad_data(reversal_ivsa_exp, False, False)