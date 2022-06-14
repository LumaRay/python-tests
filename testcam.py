import cv2

import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = 'rtsp_transport;udp | protocol_whitelist;file,rtp,udp,tcp | fflags;nobuffer | flag;low_delay'

print(cv2.getBuildInformation())

#vcap = cv2.VideoCapture("rtsp://admin:LABCC0805%24@192.168.7.147", cv2.CAP_FFMPEG)#/Streaming/Channels/102")
#vcap = cv2.VideoCapture("rtsp://admin:LABCC0805%24@192.168.1.64", cv2.CAP_DSHOW)#/Streaming/Channels/102")
#vcap = cv2.VideoCapture("rtsp://admin:LABCC0805%24@192.168.1.64", cv2.CAP_MSMF)#/Streaming/Channels/102")
#vcap = cv2.VideoCapture("rtsp://admin:LABCC0805%24@192.168.1.64", cv2.CAP_INTEL_MFX)#/Streaming/Channels/102")
#rtspGstreamer = "rtspsrc location=" + rtspString + " latency=0 buffer-mode=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink"
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! rtph264depay ! h264parse ! avdec_h264 ! videorate ! video/x-raw, framerate=(fraction)15/1 ! videoconvert ! appsink max-buffers=1 drop=True", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc framerate=10 location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! rtph264depay ! h264parse ! omxh264dec enable-low-outbuffer=1 ! videorate ! video/x-raw, framerate=(fraction)10/1 ! videoconvert ! appsink max-buffers=1 drop=True", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc framerate=25 location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! rtph264depay ! h264parse ! omxh264dec enable-low-outbuffer=1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videorate ! video/x-raw, framerate=(fraction)25/1 ! videoconvert ! appsink max-buffers=1 drop=True", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=10 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=(string)BGRx ! videorate ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture("rtspsrc framerate=25 location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! video/x-raw, format=(string)BGRx ! videorate ! video/x-raw, framerate=(fraction)25/1 ! videoconvert ! appsink max-buffers=1 drop=True", cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture(("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink").format(1920, 1080), cv2.CAP_GSTREAMER)
#vcap = cv2.VideoCapture(("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True").format(1920, 1080), cv2.CAP_GSTREAMER)

#vcap = cv2.VideoCapture(("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True").format(1920, 1080), cv2.CAP_GSTREAMER)

# sudo systemctl restart nvargus-daemon
#vcap.set(cv2.CAP_PROP_FPS, 100)
#vcap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#vcap.set(cv2.CAP_PROP_POS_FRAMES, 1)

#vcap = cv2.VideoCapture("rtsp://admin:LABCC0805%24@192.168.7.147", cv2.CAP_FFMPEG)
#vcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

while(1):

    ret, frame = vcap.read()
    if not ret:
        continue
    cv2.imshow('VIDEO', frame)
    cv2.waitKey(1)

'''from hikvisionapi import Client

cam = Client('http://192.168.1.64', 'admin', 'LABCC0805$')

response = cam.System.deviceInfo(method='get')

response = cam.System.deviceInfo(method='get', present='text')

motion_detection_info = cam.System.Video.inputs.channels[1].motionDetection(method='get')

response = cam.Streaming.channels[102].picture(method='get', type='opaque_data')
with open('screen.jpg', 'wb') as f:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)'''