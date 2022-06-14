import pathlib

import numpy as np

SCRIPT_FOLDER_PATH = str(pathlib.Path().absolute())
filename = SCRIPT_FOLDER_PATH + "/memmap.npy"
# filename = SCRIPT_FOLDER_PATH + "/memmap_small.npy"
np_memmap = np.memmap(filename, dtype=np.float32, mode='r+')
# np_memmap = np.require(np.memmap(filename, dtype=np.float32, mode='r+'), requirements=['O'])  # copies all array into ram
np_memmap.resize(np_memmap.shape[0] // 512, 512)
for idx in range(5, np_memmap.shape[0] - 1):
    np_memmap[idx, ...] = np_memmap[idx + 1, ...]
# np_memmap[5:-1, :] = np_memmap[6:, :]  # temporarily copies all array into ram
np_memmap.resize(np_memmap.shape[0] - 1, 512, refcheck=False)
# np_memmap.base.resize((np_memmap.shape[0] - 1, 512))
np_memmap.flush()
# np_memmap = np.memmap(filename, dtype=np.float32, mode='r+', shape=(np_memmap.shape[0] - 1, np_memmap.shape[1]), order='C')
np_memmap = np.require(np.memmap(filename, dtype=np.float32, mode='r+', shape=(np_memmap.shape[0] - 1, np_memmap.shape[1]), order='C'), requirements=['O'])
np_memmap.flush()
pass

