import os
import random

FOLDERS_TO_PROCESS = [
    'warning! use with care!',
]

NUMBER_OF_FILES_TO_LEAVE = 5000000

for folder in FOLDERS_TO_PROCESS:
    file_list = os.listdir(folder)
    files_count = len(file_list)
    leave_idxs = random.sample(range(0, files_count), int(NUMBER_OF_FILES_TO_LEAVE))
    for f_idx, f in enumerate(file_list):
        if f_idx not in leave_idxs:
            os.remove(f)
