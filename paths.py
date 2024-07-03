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


contigency_flip_ctrl = list_files(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/CASK/reversal/ctrl', direct_access=False)

contigency_flip_cask = list_files(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/CASK/reversal/cask', direct_access=False)

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

fr1_cask_csvs = list_files(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/CASK/FR1/cask', direct_access=False)

fr1_ivsa = list_files(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/CD1 IVSA/FR1', direct_access=False)
reversal_ivsa = list_files(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/CD1 IVSA/Reversal', direct_access=False)

fr1_fent = list_files(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/mPFC/Fentanyl Tx/FR1', direct_access=True)
reversal_fent = list_files(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/mPFC/Fentanyl Tx/Reversal', direct_access=True)

fr1_veh = list_files(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/mPFC/Vehicle Tx/FR1', direct_access=True)
reversal_veh = list_files(root='/home/ftlab/Desktop/For_Andy/behavior data integrated/mPFC/Vehicle Tx/Reversal', direct_access=True)
