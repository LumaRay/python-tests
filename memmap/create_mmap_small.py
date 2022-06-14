import pathlib

import numpy as np

SCRIPT_FOLDER_PATH = str(pathlib.Path().absolute())
filename = SCRIPT_FOLDER_PATH + "/memmap_small.npy"
# filename_owned = SCRIPT_FOLDER_PATH + "/memmap_small_owned.npy"
np_memmap = np.memmap(filename, dtype=np.float32, mode='w+', shape=(1_000, 512))
# np_memmap_owned = np.require(np.memmap(filename_owned, dtype=np.float32, mode='w+', shape=(1_000, 512)), requirements=['O'])  # copies all array into ram
np_rand = np.random.rand(1_000, 512).astype(np.float32)
np_memmap[...] = np_rand
# np_memmap_owned[...] = np_rand

np_memmap.flush()
# np_memmap_owned.flush()
pass

