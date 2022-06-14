import pafy, cv2, numpy as np, cupy as cp

'''from mtcnn import MTCNN
detector = MTCNN()'''

#pip3 install cupy-cuda101
url = 'https://www.youtube.com/watch?v=xn7wPPSh6yI'
#url = 'https://youtu.be/SxIUyECUEik'
#url = 'https://youtu.be/3LwWl2wU4tQ'
#url = 'https://youtu.be/uSLZfNteDxM'
vPafy = pafy.new(url)
play = vPafy.getbest()#(preftype="webm")

import matplotlib.pyplot as plt
import math

'''def hype(x):
    y = np.arctan(x * 10) / (0.94 * math.pi / 2)
    return y'''
'''x = np.linspace(-10, 10) / 10
y = hype(x)
plt.plot(x, y)
plt.show()'''

#start the video
cap = cv2.VideoCapture(play.url)
#cap = cv2.VideoCapture('C:\\Users\\Jure\\Downloads\\200506_kitchen Food_02_4k_007.mp4')
#cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

lastframe = None
lastframe_gpu = None
frames_gpu_queue = []

while (True):
    ret, frame = cap.read()
    """
    your code here
    """
    '''if lastframe is None:
        lastframe = frame
        continue'''

    '''frame_int32 = frame.astype(np.int32)
    lastframe_int32 = lastframe.astype(np.int32)
    diff_int32 = frame_int32 - lastframe_int32
    diff_k = diff_int32 / lastframe_int32
    diff_k3 = 10 * diff_k
    diff3_float32 = diff_k3 * lastframe_int32
    newframe3_float32 = lastframe_int32 + diff3_float32
    #newframe3_float32[newframe3_float32 > 255] = 255
    #newframe3_float32[newframe3_float32 < 0] = 0
    newframe3 = newframe3_float32.astype(np.uint8)'''

    #newframe3 = lastframe + (frame - lastframe) * 10

    '''diff = frame.astype(np.float) - lastframe.astype(np.float)
    diff_norm = diff / 255
    diff_norm_hype = y = np.arctan(diff_norm * 10) / (0.94 * math.pi / 2)
    diff_hype = diff_norm_hype * 255
    newframe3 = lastframe + diff_hype.astype(np.uint8)'''

    #frame = cv2.GaussianBlur(frame, (5, 5), 0)

    '''frame_gpu = cp.asarray(frame)
    if lastframe_gpu is None:
        lastframe_gpu = frame_gpu
        continue
    diff_gpu = frame_gpu.astype(cp.float) - lastframe_gpu.astype(np.float)
    #diff_norm_gpu = diff_gpu / 255
    #diff_norm_hype_gpu = cp.arctan(diff_norm_gpu * 10) / (0.94 * math.pi / 2)
    #diff_hype_gpu = diff_norm_hype_gpu * 255
    diff_hype_gpu = cp.arctan(diff_gpu * (10 / 255)) / (0.94 * math.pi / (2 * 255))
    newframe3_gpu = lastframe_gpu + diff_hype_gpu.astype(cp.uint8)
    newframe3 = cp.asnumpy(newframe3_gpu)
    lastframe_gpu = frame_gpu'''

    '''hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)
    #val = val * 2
    val = np.arctan(val * (10 / 255)) / (0.94 * math.pi / (2 * 255))
    val = val.astype(np.uint8)
    #hsv = cv2.merge((hue, sat, val))
    #newframe3 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    newframe3 = cv2.cvtColor(val, cv2.COLOR_GRAY2RGB)'''

    '''hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    _, _, val = cv2.split(hsv)
    val = cv2.GaussianBlur(val, (5, 5), 0)
    frames_gpu_queue.append(cp.asarray(val))
    if len(frames_gpu_queue) > 8:
        frame_min_gpu = None
        frames_gpu_queue.pop(0)
        for frame_gpu in frames_gpu_queue:
            if frame_min_gpu is None:
                frame_min_gpu = frame_gpu
                continue
            frame_min_gpu = cp.minimum(frame_min_gpu, frame_gpu)
        #val_gpu = cp.arctan(frame_min_gpu * (10 / 255)) / (0.94 * math.pi / (2 * 255))
        #val_gpu = val_gpu.astype(np.uint8)
        val_gpu = frame_min_gpu.astype(np.uint8)
        val = cp.asnumpy(val_gpu)
        newframe3 = cv2.cvtColor(val, cv2.COLOR_GRAY2RGB)
    else:
        newframe3 = frame'''

    '''image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    if len(result) > 0:
        # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
        bounding_box = result[0]['box']
        keypoints = result[0]['keypoints']
        cv2.rectangle(image,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 155, 255),
                      2)
        cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
        cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
        cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
        cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
        cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    newframe3 = image'''

    #cv2.imshow('frame', frame)

    cv2.imshow('frame', newframe3)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()