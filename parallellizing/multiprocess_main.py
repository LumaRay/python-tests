import time

from multiprocess_external import runProcess
from multiprocessing.spawn import freeze_support

def processLoadCPU(no):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # import psutil
    # import os
    # p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    # Get or set process niceness (priority). On UNIX this is a number which usually goes from -20 to 20
    # The higher the nice value, the lower the priority of the process.
    # p.nice(20)
    start_time = round(time.monotonic() * 1000)
    while True:
        time.sleep(0.001)
        # try:
        time_taken_idle = round(time.monotonic() * 1000) - start_time
        # print(no, "processLoadCPU loop idle", str(time_taken_idle), "ms")  # 10 ms
        start_time = round(time.monotonic() * 1000)
        # lst = [i for i in range(1000000)]
        # lst = [i * i for i in lst]
        # lst = [i * i for i in lst]
        # lst = lst
        time_taken = round(time.monotonic() * 1000) - start_time
        print(no, "processLoadCPU loop", time_taken, "+", time_taken_idle, "ms")
        # with youtube, no realsense 2 loads 1460 on 10000000 GPU=28% CPU=130%
        # with youtube, realsense 2 loads 1460 on 10000000 GPU=24% CPU=115%
        # without youtube, no realsense 2 loads 1330 on 10000000 GPU=12% CPU=100%
        # without youtube, no realsense 1 load 1290 on 10000000 GPU=3% CPU=100%
        start_time = round(time.monotonic() * 1000)
        # except KeyboardInterrupt:
        #     # **** THIS PART NEVER EXECUTES. ****
        #     # pool.terminate()
        #     print("You cancelled the program!")
        #     # sys.exit(1)
        #     # return

if __name__ == '__main__':
    freeze_support()
    runProcess(processLoadCPU, args=(1,))
    time.sleep(1000.001)
