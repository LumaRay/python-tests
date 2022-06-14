import cv2

import numpy as np

import time

print(cv2.cuda.getCudaEnabledDeviceCount())

npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
cuMat = cv2.cuda_GpuMat()
cuMat.upload(npMat)

print(np.allclose(cuMat.download(), npMat))

h_frame = (np.random.random((1080, 1920, 3)) * 255).astype(np.uint8)
g_frame = cv2.cuda_GpuMat()
starttime = time.process_time()
g_frame.upload(h_frame)
print(time.process_time() - starttime)
starttime = time.process_time()
g_frame.upload(h_frame)
print(time.process_time() - starttime)
starttime = time.process_time()
g_frame.upload(h_frame)
print(time.process_time() - starttime)
starttime = time.process_time()
g_frame = cv2.cuda.resize(g_frame, (25500, 14400))
print(time.process_time() - starttime)
starttime = time.process_time()
res = g_frame.download()
print(time.process_time() - starttime)
starttime = time.process_time()
h_frame = cv2.resize(h_frame, (25500, 14400))
print(time.process_time() - starttime)
#print(np.allclose(res, h_frame))
