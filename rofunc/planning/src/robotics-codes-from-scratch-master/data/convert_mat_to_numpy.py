from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
DATASET_LETTERS_FOLDER = "2Dletters/"

dataset_folder = Path(FILE_PATH,DATASET_LETTERS_FOLDER)
file_list = [str(p) for p in dataset_folder.rglob('*') if p.is_file() and p.match('*.mat')]

for file_name in file_list:
    junk_array = scipy.io.loadmat(file_name)["demos"].flatten()
    
    letter_demos = []
    
    for demo in junk_array:
        pos = demo[0]['pos'].flatten()[0].T
        vel = demo[0]['vel'].flatten()[0].T
        acc = demo[0]['acc'].flatten()[0].T
        
        demo_concatenated = np.hstack(( pos , vel , acc ))
        letter_demos += [demo_concatenated]
        
    letter_demos = np.asarray(letter_demos)
    file_name_npy = Path( Path(file_name).parent , Path(file_name).stem + ".npy") 
    np.save(file_name_npy,letter_demos)