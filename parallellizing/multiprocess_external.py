from multiprocessing import Process
import threading


def runProcess(func, args):
    # procs = [Process(target=func, args=args) for i in range(2)]
    # [p.start() for p in procs]
    # return func(args,)
    def run(func, args):
        proc = Process(target=func, args=args)
        proc.start()
    hThread = threading.Thread(target=run, args=(func, args))
    hThread.start()
