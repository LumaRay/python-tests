import time

import cv2

from pafy import pafy
# youtube_dl -> yt-dlp
# sudo pip3 install yt-dlp
# /home/thermalview/.local/lib/python3.6/site-packages/pafy/backend_youtube_dl.py
# import youtube_dl -> import yt_dlp as youtube_dl
# #self._dislikes = self._ydl_info['dislike_count']

'''vPafy = pafy.new("https://www.youtube.com/watch?v=HC9FDfuUpKQ")
play = vPafy.getbest()  # (preftype="webm")
capNormal = cv2.VideoCapture(play.url)
while True:
    time.sleep(0.010)
    start_time = round(time.monotonic() * 1000)
    ret, frameNormal = capNormal.read()
    time_taken = round(time.monotonic() * 1000) - start_time
    print("frame read time", str(time_taken), "ms")
    cv2.imshow("123", frameNormal)
    cv2.waitKey(1)'''

import numpy as np

import pyrealsense2 as rs

import threading

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







def threadCaptureYoutube():
    vPafy = pafy.new("https://www.youtube.com/watch?v=HC9FDfuUpKQ")
    play = vPafy.getbest()  # (preftype="webm")
    capNormal = cv2.VideoCapture(play.url)
    start_time = round(time.monotonic() * 1000)
    while True:
        time.sleep(0.010)
        # time.sleep(0.050)
        # time.sleep(0.500)
        time_taken = round(time.monotonic() * 1000) - start_time
        print("threadCaptureYoutube loop idle", str(time_taken), "ms") # 10 ms
        start_time = round(time.monotonic() * 1000)
        ret, frameNormal = capNormal.read()
        time_taken = round(time.monotonic() * 1000) - start_time
        print("threadCaptureYoutube loop", str(time_taken), "ms")
        # youtube_dl
        # without realsense 1-220-470 ms 0.500 pause=(CPU=14% GPU=7%) 0.050 pause=(CPU=20% GPU=20%) no pause=(CPU=20% GPU=11%)
        # yt_dlp
        # with realsense 1 ms 0.500 pause=(CPU=25% GPU=30%) 0.050 pause=(CPU=20% GPU=24%) no pause=(CPU=20% GPU=38%)
        # without realsense 1 ms 0.500 pause=(CPU=13% GPU=4%) 0.050 pause=(CPU=40% GPU=20%) no pause=(CPU=380% GPU=35%)
        # without realsense no imshow no pause=(CPU=460% GPU=30%)
        cv2.imshow("123", frameNormal)
        cv2.waitKey(1)
        start_time = round(time.monotonic() * 1000)




def threadCaptureRealsense():
    start_time = round(time.monotonic() * 1000)
    while True:
        time.sleep(0.010)
        time_taken = round(time.monotonic() * 1000) - start_time
        # print("threadCaptureRealsense loop idle", str(time_taken), "ms") # 10 ms
        start_time = round(time.monotonic() * 1000)
        frames = realsense_pipe.wait_for_frames()

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
        print("threadCaptureRealsense loop", str(time_taken), "ms") # 23 ms
        start_time = round(time.monotonic() * 1000)

def threadLoadCPU(no):
    start_time = round(time.monotonic() * 1000)
    while True:
        time.sleep(0.010)
        time_taken_idle = round(time.monotonic() * 1000) - start_time
        # print(no, "threadLoadCPU loop idle", str(time_taken_idle), "ms") # 10 ms
        start_time = round(time.monotonic() * 1000)
        lst = [i for i in range(1000000)]
        lst = [i*i for i in lst]
        lst = [i*i for i in lst]
        time_taken = round(time.monotonic() * 1000) - start_time
        print(no, "threadLoadCPU loop", time_taken, "+", time_taken_idle, "ms")
        start_time = round(time.monotonic() * 1000)



hCaptureYoutubeThread = threading.Thread(target=threadCaptureYoutube)
hCaptureRealsenseThread = threading.Thread(target=threadCaptureRealsense)
hLoadCPUThread1 = threading.Thread(target=threadLoadCPU, args=(1,))
hLoadCPUThread2 = threading.Thread(target=threadLoadCPU, args=(2,))

# hCaptureYoutubeThread.start()
# hCaptureRealsenseThread.start()
hLoadCPUThread1.start()
hLoadCPUThread2.start()
