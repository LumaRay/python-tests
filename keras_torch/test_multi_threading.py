import threading
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from face_recognition_modules.deepface.DeepFace import build_model

TENSORFLOW_GPU_MEMORY_LIMIT_MB = 2 * 1024  # 3
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(gpu, [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=TENSORFLOW_GPU_MEMORY_LIMIT_MB)])
    except RuntimeError as e:
        print(e)

uem_mask_model = load_model(
    "/home/thermalview/Desktop/ThermalView/mask_detection/uem_mask/ta946tl180_i_va0.9989517vl0.0014108_r01e0001_allw12vk2p1randd5v20-1-20-0-128l256l512l728l1024-128x128x3-sn.h5")


def threadUemMask():
    in_frames_array = np.zeros((1, 128, 128, 3), dtype=np.uint8)
    predictions = uem_mask_model.predict(in_frames_array)
    start_time = round(time.monotonic() * 1000)
    while True:
        time.sleep(0.0010)
        time_taken_idle = round(time.monotonic() * 1000) - start_time
        # print("uem_mask_model.predict loop idle", str(time_taken_idle), "ms")  #
        start_time = round(time.monotonic() * 1000)
        predictions = uem_mask_model.predict(in_frames_array)
        time_taken = round(time.monotonic() * 1000) - start_time
        print("uem_mask_model.predict loop", time_taken, "+", time_taken_idle, "=", time_taken + time_taken_idle, "ms")  # 33-57
        start_time = round(time.monotonic() * 1000)

def threadFacenet512():
    facenet512_model = build_model("Facenet512")
    # facenet512_model = Facenet512.loadModel("/home/thermalview/Desktop/ThermalView/face_recognition_modules/deepface/weights/facenet512_weights.h5")
    # facenet512_model = facenet512_model()
    # Facenet512.loadModel("/home/thermalview/Desktop/ThermalView/face_recognition_modules/deepface/weights/facenet512_weights.h5")
    in_frames_array = np.zeros((1, 160, 160, 3), dtype=np.uint8)
    predictions = facenet512_model.predict(in_frames_array)
    start_time = round(time.monotonic() * 1000)
    while True:
        time.sleep(0.0010)
        time_taken_idle = round(time.monotonic() * 1000) - start_time
        start_time = round(time.monotonic() * 1000)
        predictions = facenet512_model.predict(in_frames_array)
        time_taken = round(time.monotonic() * 1000) - start_time
        print("facenet512_model.predict loop", time_taken, "+", time_taken_idle, "=", time_taken + time_taken_idle, "ms")  # 52-55
        start_time = round(time.monotonic() * 1000)

hUemMaskThread = threading.Thread(target=threadUemMask)
hFacenet512Thread = threading.Thread(target=threadFacenet512)

hUemMaskThread.start()
hFacenet512Thread.start()

hUemMaskThread.join()
hFacenet512Thread.join()

# RAM8.9 GPURAM3 GPU22 CPU116

'''uem_mask_model.predict loop 41 + 1 = 42 ms
facenet512_model.predict loop 50 + 2 = 52 ms
uem_mask_model.predict loop 34 + 1 = 35 ms
facenet512_model.predict loop 52 + 1 = 53 ms
uem_mask_model.predict loop 41 + 1 = 42 ms
uem_mask_model.predict loop 42 + 1 = 43 ms
facenet512_model.predict loop 63 + 1 = 64 ms
uem_mask_model.predict loop 32 + 2 = 34 ms
facenet512_model.predict loop 52 + 3 = 55 ms
uem_mask_model.predict loop 41 + 1 = 42 ms
uem_mask_model.predict loop 53 + 1 = 54 ms
facenet512_model.predict loop 56 + 1 = 57 ms
uem_mask_model.predict loop 56 + 1 = 57 ms
facenet512_model.predict loop 55 + 2 = 57 ms
uem_mask_model.predict loop 52 + 1 = 53 ms
facenet512_model.predict loop 59 + 1 = 60 ms
uem_mask_model.predict loop 42 + 1 = 43 ms
facenet512_model.predict loop 51 + 2 = 53 ms
uem_mask_model.predict loop 35 + 1 = 36 ms
facenet512_model.predict loop 50 + 1 = 51 ms
uem_mask_model.predict loop 34 + 1 = 35 ms
facenet512_model.predict loop 54 + 1 = 55 ms
uem_mask_model.predict loop 51 + 1 = 52 ms
uem_mask_model.predict loop 45 + 1 = 46 ms
facenet512_model.predict loop 57 + 1 = 58 ms
uem_mask_model.predict loop 31 + 1 = 32 ms
facenet512_model.predict loop 68 + 3 = 71 ms
uem_mask_model.predict loop 52 + 1 = 53 ms
facenet512_model.predict loop 57 + 1 = 58 ms
uem_mask_model.predict loop 56 + 1 = 57 ms
uem_mask_model.predict loop 52 + 1 = 53 ms
facenet512_model.predict loop 55 + 1 = 56 ms
uem_mask_model.predict loop 49 + 1 = 50 ms
facenet512_model.predict loop 56 + 2 = 58 ms
uem_mask_model.predict loop 56 + 1 = 57 ms
facenet512_model.predict loop 54 + 1 = 55 ms
uem_mask_model.predict loop 44 + 1 = 45 ms
facenet512_model.predict loop 49 + 1 = 50 ms
uem_mask_model.predict loop 37 + 1 = 38 ms
facenet512_model.predict loop 48 + 1 = 49 ms
uem_mask_model.predict loop 33 + 1 = 34 ms
facenet512_model.predict loop 54 + 1 = 55 ms
uem_mask_model.predict loop 44 + 1 = 45 ms
uem_mask_model.predict loop 54 + 1 = 55 ms
facenet512_model.predict loop 57 + 1 = 58 ms
facenet512_model.predict loop 52 + 2 = 54 ms
uem_mask_model.predict loop 55 + 1 = 56 ms
facenet512_model.predict loop 54 + 2 = 56 ms
uem_mask_model.predict loop 52 + 1 = 53 ms
uem_mask_model.predict loop 50 + 1 = 51 ms
facenet512_model.predict loop 60 + 1 = 61 ms
uem_mask_model.predict loop 41 + 1 = 42 ms
facenet512_model.predict loop 54 + 2 = 56 ms
uem_mask_model.predict loop 42 + 1 = 43 ms
facenet512_model.predict loop 52 + 1 = 53 ms
uem_mask_model.predict loop 41 + 1 = 42 ms
facenet512_model.predict loop 52 + 1 = 53 ms
uem_mask_model.predict loop 46 + 1 = 47 ms
uem_mask_model.predict loop 47 + 1 = 48 ms
facenet512_model.predict loop 57 + 1 = 58 ms
uem_mask_model.predict loop 42 + 1 = 43 ms
facenet512_model.predict loop 51 + 2 = 53 ms
uem_mask_model.predict loop 35 + 1 = 36 ms
facenet512_model.predict loop 52 + 1 = 53 ms
uem_mask_model.predict loop 36 + 1 = 37 ms
uem_mask_model.predict loop 42 + 1 = 43 ms
facenet512_model.predict loop 66 + 1 = 67 ms
uem_mask_model.predict loop 34 + 2 = 36 ms
facenet512_model.predict loop 53 + 3 = 56 ms
uem_mask_model.predict loop 42 + 1 = 43 ms
uem_mask_model.predict loop 46 + 1 = 47 ms
facenet512_model.predict loop 60 + 1 = 61 ms
uem_mask_model.predict loop 38 + 1 = 39 ms
facenet512_model.predict loop 50 + 1 = 51 ms
uem_mask_model.predict loop 33 + 2 = 35 ms
facenet512_model.predict loop 50 + 1 = 51 ms
uem_mask_model.predict loop 43 + 1 = 44 ms
facenet512_model.predict loop 52 + 1 = 53 ms
uem_mask_model.predict loop 50 + 2 = 52 ms
facenet512_model.predict loop 56 + 1 = 57 ms
uem_mask_model.predict loop 53 + 1 = 54 ms
uem_mask_model.predict loop 48 + 2 = 50 ms
facenet512_model.predict loop 58 + 1 = 59 ms
uem_mask_model.predict loop 42 + 1 = 43 ms
facenet512_model.predict loop 51 + 1 = 52 ms
uem_mask_model.predict loop 34 + 1 = 35 ms
facenet512_model.predict loop 62 + 1 = 63 ms
uem_mask_model.predict loop 49 + 1 = 50 ms
uem_mask_model.predict loop 55 + 1 = 56 ms
facenet512_model.predict loop 58 + 1 = 59 ms
uem_mask_model.predict loop 49 + 1 = 50 ms
facenet512_model.predict loop 58 + 2 = 60 ms
uem_mask_model.predict loop 40 + 1 = 41 ms
facenet512_model.predict loop 52 + 1 = 53 ms
uem_mask_model.predict loop 34 + 1 = 35 ms
facenet512_model.predict loop 71 + 1 = 72 ms
uem_mask_model.predict loop 57 + 1 = 58 ms
uem_mask_model.predict loop 56 + 2 = 58 ms
facenet512_model.predict loop 63 + 1 = 64 ms
uem_mask_model.predict loop 42 + 1 = 43 ms
facenet512_model.predict loop 52 + 1 = 53 ms
uem_mask_model.predict loop 37 + 1 = 38 ms
facenet512_model.predict loop 50 + 1 = 51 ms
uem_mask_model.predict loop 34 + 1 = 35 ms
facenet512_model.predict loop 52 + 1 = 53 ms
uem_mask_model.predict loop 47 + 1 = 48 ms
facenet512_model.predict loop 51 + 1 = 52 ms
uem_mask_model.predict loop 50 + 1 = 51 ms
uem_mask_model.predict loop 53 + 2 = 55 ms
facenet512_model.predict loop 56 + 1 = 57 ms
uem_mask_model.predict loop 48 + 1 = 49 ms
facenet512_model.predict loop 56 + 2 = 58 ms
uem_mask_model.predict loop 41 + 1 = 42 ms
facenet512_model.predict loop 56 + 1 = 57 ms
uem_mask_model.predict loop 36 + 2 = 38 ms
facenet512_model.predict loop 57 + 3 = 60 ms
uem_mask_model.predict loop 46 + 2 = 48 ms
facenet512_model.predict loop 55 + 1 = 56 ms
uem_mask_model.predict loop 51 + 1 = 52 ms
uem_mask_model.predict loop 45 + 2 = 47 ms
facenet512_model.predict loop 57 + 1 = 58 ms
uem_mask_model.predict loop 39 + 1 = 40 ms
facenet512_model.predict loop 56 + 1 = 57 ms
uem_mask_model.predict loop 31 + 1 = 32 ms
uem_mask_model.predict loop 53 + 1 = 54 ms
facenet512_model.predict loop 57 + 2 = 59 ms
uem_mask_model.predict loop 42 + 1 = 43 ms
facenet512_model.predict loop 63 + 1 = 64 ms
uem_mask_model.predict loop 33 + 1 = 34 ms
facenet512_model.predict loop 54 + 2 = 56 ms'''