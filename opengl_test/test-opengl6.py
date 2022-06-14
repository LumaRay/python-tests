import cv2
import acapture
import pyglview
# https://pypi.org/project/pyglview/
# sudo apt install -y build-essential
# sudo apt install -y libgtkglext1 libgtkglext1-dev
# sudo apt install -y libgl1-mesa-dev libglu1-mesa-dev mesa-utils
# sudo apt install -y freeglut3-dev libglew1.10 libglew-dev libgl1-mesa-glx libxmu-dev
# sudo apt install -y libglew-dev libsdl2-dev libsdl2-image-dev libglm-dev libfreetype6-dev
# sudo pip3 install PyOpenGL PyOpenGL_accelerate
# sudo pip3 install pyglview
# sudo pip3 install acapture
# sudo apt install -y ffmpeg
# sudo apt-get install python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev
# sudo pip3 install pygame

viewer = pyglview.Viewer()
cap = acapture.open(0)  # Camera 0,  /dev/video0
def loop():
    check, frame = cap.read()  # non-blocking
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if check:
        viewer.set_image(frame)
viewer.set_loop(loop)
viewer.start()