import time

import cv2
# https://issueexplorer.com/issue/opencv/opencv/11220
# https://forums.developer.nvidia.com/t/video-mapping-on-jetson-tx1/44797#4979740
# https://gstreamer.freedesktop.org/documentation/opengl/glimagesink.html?gi-language=c
# https://coderoad.ru/9750317/python-%D0%9F%D0%BB%D0%B0%D0%B3%D0%B8%D0%BD%D1%8B-gstreamer
# https://forums.developer.nvidia.com/t/gstreamer-nvmm-opencv-gpumat/52575/6
# https://github.com/opencv/opencv/issues/11220
# nvvidconv
# nveglglessink
# nveglglesink
# nvivafilter
# /home/thermalview/Downloads/X2Convert.com what_language_do_you_find_most_attractive_street_interviews_-8052641072221435869.mp4
# gst-launch-1.0 filesrc location=<my_movie.mp4> ! qtdemux ! h264parse ! omxh264dec ! nvvidconv ! ‘video/x-raw, format=(string)RGBA’ ! glimagesink
# gst-launch-1.0 filesrc location=<my_movie.mp4> ! qtdemux ! h264parse ! omxh264dec ! nvvidconv ! ‘video/x-raw, format=(string)RGBA’ ! tee ! glimagesink
# gst-launch-1.0 filesrc location=~/b.mp4 ! qtdemux ! h264parse ! omxh264dec ! nvvidconv ! "video/x-raw(memory:NVMM),format=(string)NV12" ! nvegltransform ! "video/x-raw(memory:EGLImage),format=(string)RGBA" ! nveglglessink window-x=100 window-y=100
# gst-launch-1.0 filesrc location=<filename.mp4> ! qtdemux name=demux ! h264parse ! omxh264dec ! nvivafilter customer-lib-name=libnvsample_cudaprocess.so cuda-process=true post-process=true ! "video/x-raw(memory:NVMM),format=(string)RGBA" ! nvegltransform ! nveglglessink -e
# gst-launch-1.0 filesrc location=<filename.mp4> ! qtdemux name=demux ! h264parse ! omxh264dec ! nvivafilter customer-lib-name=libnvsample_cudaprocess.so cuda-process=true post-process=true ! "video/x-raw(memory:NVMM),format=(string)RGBA" ! nvoverlaysink display-id=1 -e
# ПРЕДУПРЕЖДЕНИЕ: ошибочный конвейер: элемент «omxh264dec» не найден
# gst-launch-1.0 filesrc location=<my_file.mp4> ! qtdemux ! h264parse ! omxh264dec ! glimagesink
# gst-launch-1.0 filesrc location= ~/Bourne_Trailer.mp4 ! qtdemux name=demux ! h264parse ! omxh264dec ! nvivafilter customer-lib-name=libnvsample_cudaprocess.so cuda-process=true ! 'video/x-raw(memory:NVMM),format=RGBA' ! nvegltransform ! nveglglessink
# gst-launch-1.0 v4l2src ! "video/x-raw, format=(string)UYVY, width=(int)3840,height=(int)2160" ! nvvidconv ! "video/x-raw(memory:NVMM),format=(string)I420, width=(int)3840, height=(int)2160" ! nvoverlaysink overlay-w=1920 overlay-h=1080 sync=false
# gst-launch-1.0 v4l2src ! "video/x-raw, format=(string)UYVY, width=(int)3840,height=(int)2160" ! nvvidconv ! 'video/x-raw(memory:NVMM),format=(string)I420, width=(int)3840, height=(int)2160' ! nvivafilter customer-lib-name=./lib-gst-custom-opencv_cudaprocess.so cuda-process=true ! 'video/x-raw(memory:NVMM), format=RGBA' ! nvoverlaysink overlay-w=1920 overlay-h=1080 sync=false
# gst-launch-1.0 nvcamerasrc fpsRange="30 30" sensor-id=1 ! 'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)I420, framerate=(fraction)30/1' ! nvtee ! nvivafilter cuda-process=true customer-lib-name=./lib-gst-custom-opencv_cudaprocess.so ! 'video/x-raw(memory:NVMM), format=(string)NV12' ! nvegltransform ! nveglglessink
# gst-launch-1.0 nvcamerasrc ! 'video/x-raw(memory:NVMM), format=I420' ! nvivafilter customer-lib-name="./lib-gst-custom-opencv_cudaprocess.so" cuda-process=true ! 'video/x-raw(memory:NVMM), format=RGBA' ! nvegltransform ! nveglglessink
# gst-launch-1.0 nvcamerasrc fpsRange="30 30" sensor-id=1 ! 'video/x-raw(memory:NVMM), width=(int)1920, height=1080, format=(string)I420, framerate=(fraction)30/1' ! nvtee ! nvivafilter cuda-process=true customer-lib-name=./lib-gst-custom-opencv_cudaprocess.so ! 'video/x-raw(memory:NVMM), format=(string)RGBA' ! nvegltransform ! nveglglessink
# gst-launch-1.0 nvcamerasrc fpsRange="30 30" sensor-id=1 ! 'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)I420, framerate=(fraction)30/1' ! nvtee ! nvivafilter cuda-process=true customer-lib-name=./lib-gst-custom-opencv_cudaprocess.so ! 'video/x-raw(memory:NVMM), format=(string)RGBA' ! nvegltransform ! nveglglessink
# gst-launch-1.0 nvcamerasrc ! 'video/x-raw(memory:NVMM)' ! nvivafilter customer-lib-name=libnvsample_cudaprocess.so cuda-process=true ! 'video/x-raw(memory:NVMM),format=(string)RGBA' ! nvoverlaysink
# gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw,width=1920,height=1080,format=UYVY ! nvvidconv ! video/x-raw(memory:NVMM),width=1920,height=1080,format=I420 ! nvivafilter customer-lib-name=./lib-gst-custom-opencv_cudaprocess.so cuda-process=true ! 'video/x-raw(memory:NVMM), format=RGBA' ! nvvidconv ! videoconvert ! video/x-raw, format=BGR ! appsink
# gst-launch-1.0 v4l2src device=/dev/video10 ! video/x-raw, format=UYVY, width=640, height=480, framerate=30/1 ! nvvidconv ! video/x-raw(memory:NVMM) ! nvivafilter customer-lib-name=<path_to_your_lib>/libnvsample_cudaprocess.so cuda-process=true ! video/x-raw(memory:NVMM), format=RGBA, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink
# gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw,width=1920,height=1080,format=(string)UYVY ! nvvidconv ! video/x-raw(memory:NVMM),width=1920,height=1080,format=(string)I420 ! nvvidconv! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink
# https://developer.nvidia.com/nvidia-video-codec-sdk
from pafy import pafy

fname = "0"
# fname = "/home/thermalview/Downloads/X2Convert.com what_language_do_you_find_most_attractive_street_interviews_-8052641072221435869.mp4"
# gst-launch-1.0 -v videotestsrc ! video/x-raw ! glimagesink
# gst-launch-1.0 -v videotestsrc ! video/x-raw,format=I420 ! glimagesink
# gst-launch-1.0 -v gltestsrc ! glimagesink
# OPENCV_FFMPEG_CAPTURE_OPTIONS="video_codec;h264_cuvid|rtsp_transport;tcp"
# export OPENCV_FFMPEG_CAPTURE_OPTIONS="video_codec|h264_cuvid"


# vPafy = pafy.new('https://www.youtube.com/watch?v=HC9FDfuUpKQ')
# play = vPafy.getbest()  # (preftype="webm")
# d_reader = cv2.cudacodec.createVideoReader(play.url)  # fname)
# ffplay -fflags nobuffer -flags low_delay -framedrop -strict experimental -avioflags direct -probesize 32 -rtsp_transport tcp rtsp://admin:LABCC0805%24@192.168.7.147
# rtspGstreamer = 'rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink max-buffers=1 drop=True'
rtspGstreamer = 'rtsp://admin:LABCC0805%24@192.168.7.147'
# capNormal = cv2.VideoCapture(rtspGstreamer, cv2.CAP_GSTREAMER)
d_reader = cv2.cudacodec.createVideoReader(rtspGstreamer)  # fname)
d_reader.set(cv2.CAP_PROP_POS_FRAMES, 100000)
# self.capNormal.set(cv2.CAP_PROP_POS_FRAMES, 0)
d_reader.set(cv2.CAP_PROP_BUFFERSIZE, 0)
# self.capNormal.set(cv2.CAP_PROP_POS_FRAMES, 1000)
d_reader.set(cv2.CAP_PROP_POS_MSEC, 100000)
d_reader.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
d_reader.set(cv2.CAP_PROP_FPS, 100)

print([i for i in list(cv2.__dict__.keys()) if i[:6] == 'WINDOW'])
# cv2.namedWindow("test", cv2.WINDOW_OPENGL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_EXPANDED | cv2.WINDOW_KEEPRATIO)
while True:
    # time.sleep(0.03)
    res, d_frame = d_reader.nextFrame()
    if not res or d_frame is None:
        continue
    result = d_frame.download()
    # cv2.cuda.imshow("test", d_frame)
    cv2.imshow("test", result)
    cv2.waitKey(1)
