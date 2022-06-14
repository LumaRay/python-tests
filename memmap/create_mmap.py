import pathlib

import numpy as np

SCRIPT_FOLDER_PATH = str(pathlib.Path().absolute())
filename = SCRIPT_FOLDER_PATH + "/memmap.npy"
np_memmap = np.memmap(filename, dtype=np.float32, mode='w+', shape=(1_000_000, 512))
np_rand = np.random.rand(1_000_000, 512).astype(np.float32)
np_memmap[...] = np_rand

np_memmap.flush()


