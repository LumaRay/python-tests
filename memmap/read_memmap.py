import pathlib

import numpy as np

SCRIPT_FOLDER_PATH = str(pathlib.Path().absolute())
filename = SCRIPT_FOLDER_PATH + "/memmap.npy"
np_memmap = np.memmap(filename, dtype=np.float32, mode='r+', shape=(1_000_000, 512))
np_memmap[100, 4] = np_memmap[345, 114]
np_memmap.flush()
pass
