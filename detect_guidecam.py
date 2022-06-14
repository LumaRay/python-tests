import os
from sys import platform as _platform

import subprocess

if _platform == "win32" or _platform == "win64":
    #os.system('gst-device-monitor-1.0.exe Video/Source')
    # device-path="..."
    output = subprocess.check_output("gst-device-monitor-1.0.exe Video/Source", shell=True)
    #print(output)
    output = str(output)
    index_guidecamera = output.find('GuideCamera')
    index_guidecamera_dev_start = output.find('device-path=', index_guidecamera)
    index_guidecamera_dev_end = output.find('\"', index_guidecamera_dev_start + 13)
    devpath_guidecamera = output[index_guidecamera_dev_start:index_guidecamera_dev_end + 1]
    #print(devpath_guidecamera)

else:
    #os.system('gst-device-monitor-1.0 Video/Source')
    output = subprocess.check_output("gst-device-monitor-1.0 Video/Source", shell=True)
    #print(output)
    output = str(output)
    index_guidecamera = output.find('GuideCamera')
    index_guidecamera_dev_start = output.find('device=/dev/video', index_guidecamera)
    index_guidecamera_dev_end = output.find(' ', index_guidecamera_dev_start)
    devpath_guidecamera = output[index_guidecamera_dev_start:index_guidecamera_dev_end]
    print(devpath_guidecamera)
    '''stream = os.popen('gst-device-monitor-1.0 Video/Source')
    output = stream.read()
    output'''

