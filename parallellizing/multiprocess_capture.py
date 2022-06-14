import multiprocessing as mp
from multiprocessing import Process
from multiprocessing.spawn import freeze_support
import time

import numpy as np


def processCaptureYoutube():
    import cv2
    from pafy import pafy
    vPafy = pafy.new("https://www.youtube.com/watch?v=HC9FDfuUpKQ")
    play = vPafy.getbest()  # (preftype="webm")
    capNormal = cv2.VideoCapture(play.url)
    with yt_frame_arr.get_lock():
        yt_frame = np.frombuffer(yt_frame_arr.get_obj(), dtype=np.uint8)
        yt_frame = yt_frame.reshape((720, 1280, 3))
    start_time = round(time.monotonic() * 1000)
    while True:
        time.sleep(0.010)
        # time.sleep(0.050)
        # time.sleep(0.500)
        time_taken = round(time.monotonic() * 1000) - start_time
        # print("processCaptureYoutube loop idle", str(time_taken), "ms") # 10 ms
        start_time = round(time.monotonic() * 1000)
        ret, frameNormal = capNormal.read()
        time_taken = round(time.monotonic() * 1000) - start_time
        # with yt_frame_arr.get_lock():
        yt_frame[...] = frameNormal[...]
        print("processCaptureYoutube loop", str(time_taken), "ms") # yt_dlp 1 ms
        # cv2.imshow("123", frameNormal)
        # cv2.waitKey(1)
        start_time = round(time.monotonic() * 1000)

def processShowYoutube():
    import cv2
    with yt_frame_arr.get_lock():
        yt_frame = np.frombuffer(yt_frame_arr.get_obj(), dtype=np.uint8)
        yt_frame = yt_frame.reshape((720, 1280, 3))
    while True:
        time.sleep(0.010)
        # with yt_frame_arr.get_lock():
        cv2.imshow("123", yt_frame)
        cv2.waitKey(1)

def processCaptureRealsense():
    import pyrealsense2 as rs
    import numpy as np

    realsense_cfg = rs.config()
    realsense_cfg.enable_stream(rs.stream.color, 1920, 1080)
    realsense_cfg.enable_stream(rs.stream.depth, 1280, 720)
    realsense_decimate = rs.decimation_filter(4)
    realsense_spatiate = rs.spatial_filter()
    realsense_spatiate.set_option(rs.option.holes_fill, 3)
    realsense_temporate = rs.temporal_filter()
    realsense_holate = rs.hole_filling_filter()
    realsense_depth_to_disparity = rs.disparity_transform(True)
    realsense_disparity_to_depth = rs.disparity_transform(False)
    realsense_color_map = rs.colorizer()
    realsense_align_to = rs.stream.color
    realsense_align = rs.align(realsense_align_to)
    realsense_pipe = rs.pipeline()
    realsense_profile = realsense_pipe.start(realsense_cfg)

    start_time = round(time.monotonic() * 1000)
    while True:
        time.sleep(0.010)
        time_taken = round(time.monotonic() * 1000) - start_time
        # print("processCaptureRealsense loop idle", str(time_taken), "ms") # 10 ms
        start_time = round(time.monotonic() * 1000)
        try:
            frames = realsense_pipe.wait_for_frames()
        except:
            continue

        frames = realsense_decimate.process(frames)
        frames = realsense_depth_to_disparity.process(frames)
        frames = realsense_spatiate.process(frames)
        frames = realsense_temporate.process(frames)
        frames = realsense_disparity_to_depth.process(frames)
        frames = realsense_holate.process(frames)

        frames = frames.as_frameset()
        depth_frame = frames.get_depth_frame()
        colorized_depth = np.asanyarray(realsense_color_map.colorize(depth_frame).get_data())
        lastRealSenseDepthFrame = colorized_depth
        lastRealSenseDepthData = np.asanyarray(depth_frame.get_data())
        time_taken = round(time.monotonic() * 1000) - start_time
        # print("processCaptureRealsense loop", str(time_taken), "ms") # 23 ms
        start_time = round(time.monotonic() * 1000)

def processLoadCPU(no):
    import psutil
    import os
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    # Get or set process niceness (priority). On UNIX this is a number which usually goes from -20 to 20
    # The higher the nice value, the lower the priority of the process.
    p.nice(20)
    start_time = round(time.monotonic() * 1000)
    while True:
        time.sleep(0.010)
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
    # freeze_support()

    yt_frame_arr = mp.Array('B', 720 * 1280 * 3)

    procCaptureYoutube = Process(target=processCaptureYoutube)
    procShowYoutube = Process(target=processShowYoutube)
    # procCaptureRealsense = Process(target=processCaptureRealsense)
    # procLoadCPU1 = Process(target=processLoadCPU, args=(1,))
    # procLoadCPU2 = Process(target=processLoadCPU, args=(2,))
    # procLoadCPUs = [Process(target=processLoadCPU, args=(i,)) for i in range(2)]

    procCaptureYoutube.start()
    procShowYoutube.start()
    # procCaptureRealsense.start()
    # procLoadCPU1.start()
    # procLoadCPU2.start()

    '''time.sleep(10.001)

    [p.start() for p in procLoadCPUs]'''

    procCaptureYoutube.join()
    procShowYoutube.join()

    # [p.join() for p in procLoadCPUs]

    # import cv2
    # cv2.waitKey(0)
    # time.sleep(1000.001)

