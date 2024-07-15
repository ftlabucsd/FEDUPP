import os


def list_files(root:str, direct_access=True):
    file_paths = []
    
    if direct_access:
        files = os.listdir(root)
        try:
            files.remove('.DS_Store')
        except:
            pass      
        file_paths = [os.path.join(root, file) for file in files]
    else:
        for root, dirs, files in os.walk(root):
            try:
                files.remove('.DS_Store')
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
            files.append(os.path.join(condition_path, os.listdir(condition_path)[0]))
            
    return files       
    

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


contigency_flip_ctrl = list_files(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/CASK/reversal/ctrl', direct_access=False)
contigency_flip_cask = list_files(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/CASK/reversal/cask', direct_access=False)


fr1_cask_csvs = list_files(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/CASK/FR1/cask', direct_access=False)

fr1_ivsa = find_condition(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/CD1 IVSA/', fr1=True)
reversal_ivsa = find_condition(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/CD1 IVSA/', fr1=False)

fr1_fent = list_files(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/mPFC/Fentanyl Tx/FR1', direct_access=True)
reversal_fent = list_files(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/mPFC/Fentanyl Tx/Reversal', direct_access=True)

fr1_veh = list_files(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/mPFC/Vehicle Tx/FR1', direct_access=True)
reversal_veh = list_files(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/mPFC/Vehicle Tx/Reversal', direct_access=True)

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
        fr1_ivsa_ctrl.append(i)
    else:
        fr1_ivsa_exp.append(i)