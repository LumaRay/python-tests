# sudo pip3 install joblib

import time

from joblib import Parallel, delayed

import multiprocessing as mp
from multiprocessing import freeze_support

def processLoadCPU(no):
    import psutil
    import os
    import numpy as np
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    # Get or set process niceness (priority). On UNIX this is a number which usually goes from -20 to 20
    # The higher the nice value, the lower the priority of the process.
    p.nice(20)
    last_time = np.frombuffer(last_time_arr.get_obj(), dtype=np.float32)
    start_time = round(time.monotonic() * 1000)
    while True:
        time.sleep(0.001)
        time_taken_idle = round(time.monotonic() * 1000) - start_time
        # print(no, "processLoadCPU loop idle", str(time_taken_idle), "ms")  # 10 ms
        start_time = round(time.monotonic() * 1000)
        lst = [i for i in range(1000000)]
        lst = [i * i for i in lst]
        lst = [i * i for i in lst]
        time_taken = round(time.monotonic() * 1000) - start_time
        print(no, "processLoadCPU loop", time_taken, "+", time_taken_idle, "ms")
        # with youtube, no realsense 2 loads 1460 on 10000000 GPU=28% CPU=130%
        # with youtube, realsense 2 loads 1460 on 10000000 GPU=24% CPU=115%
        # without youtube, no realsense 2 loads 1330 on 10000000 GPU=12% CPU=100%
        # without youtube, no realsense 1 load 1290 on 10000000 GPU=3% CPU=100%
        start_time = round(time.monotonic() * 1000)

if __name__ == '__main__':
    freeze_support()
    LOAD_CPU_TASKS_COUNT = 2
    last_time_arr = mp.Array('f', LOAD_CPU_TASKS_COUNT)
    Parallel(n_jobs=LOAD_CPU_TASKS_COUNT)(delayed(processLoadCPU)(i) for i in range(LOAD_CPU_TASKS_COUNT))
    # Parallel(n_jobs=1)((delayed(processLoadCPU)(0), delayed(processLoadCPU)(1)))
