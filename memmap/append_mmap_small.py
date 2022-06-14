import pathlib

import numpy as np

SCRIPT_FOLDER_PATH = str(pathlib.Path().absolute())
filename = SCRIPT_FOLDER_PATH + "/memmap_small.npy"
# filename_owned = SCRIPT_FOLDER_PATH + "/memmap_small_owned.npy"
# np_memmap = np.memmap(filename, dtype=np.float32, mode='r+', shape=(1_000, 512))
np_memmap = np.memmap(filename, dtype=np.float32, mode='r+')
np_memmap.resize(np_memmap.shape[0] // 512, 512)
# np_memmap_owned = np.require(np.memmap(filename_owned, dtype=np.float32, mode='w+', shape=(1_000, 512)), requirements=['O'])  # copies all array into ram
# np_rand = np.random.rand(1_000, 512).astype(np.float32)
np_rand_item = np.random.rand(1, 512).astype(np.float32)
# np_memmap.resize(np_memmap.shape[0] + 1, refcheck=False)
# np_memmap_owned.resize((np_memmap_owned.shape[0] + 1, np_memmap_owned.shape[1]), refcheck=False)
# np_memmap_owned.resize((np_memmap_owned.shape[0] + 1, np_memmap_owned.shape[1]), refcheck=True)
# np_memmap = np.append(np_memmap, np_rand_item, axis=0)
np_memmap = np.memmap(filename, dtype=np.float32, mode='r+', shape=(np_memmap.shape[0] + 1, np_memmap.shape[1]), order='C')
# np_memmap_placeholder[-1, ...] = np_rand_item[0, ...]
np_memmap[-1, ...] = np_rand_item[0, ...]
# np_memmap_owned[-1, ...] = np_rand_item[0, ...]
np_memmap.flush()
# np_memmap_owned.flush()
# np_memmap_placeholder.flush()
pass

