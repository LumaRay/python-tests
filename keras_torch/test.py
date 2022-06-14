face_image_path = "/home/thermalview/Desktop/ThermalView/alert/2021_12_16/faces/temperature/color/2021_12_16__12_42_43_685797_TEMP_COLOR_5.7.jpg"

import time

import imp
import torch

from facenet_pytorch import InceptionResnetV1 as TimeslerInceptionResnetV1

import cv2

import numpy as np

with torch.no_grad():
    # torch_weights = "/home/thermalview/Desktop/ThermalView/tests/keras_torch/keras_to_torch_uem_mask.pt" # path_to_the_numpy_weights
    torch_weights = "/home/thermalview/Desktop/ThermalView/tests/keras_torch/keras_to_torch_uem_mask_2.pt" # path_to_the_numpy_weights
    # A = imp.load_source('MainModel', 'keras_to_torch_uem_mask.py')
    A = imp.load_source('MainModel', 'keras_to_torch_uem_mask_2.py')

    model = torch.load(torch_weights)
    # model = torch.load(torch_weights, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    uem_mask_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(uem_mask_device)
    model.eval()

    torch_image = torch.ones([1, 3, 128, 128], dtype=torch.float32, device=uem_mask_device)

    test_img = cv2.imread(face_image_path)
    test_img_128 = cv2.resize(test_img, (128, 128))
    test_img_160 = cv2.resize(test_img, (160, 160))
    test_img_128 = np.expand_dims(test_img_128, 0)

    torch_image = torch.from_numpy(test_img_128)
    # torch_image = torch.transpose(torch_image, 1, 3)
    # torch_image = torch.transpose(torch_image, 2, 3)
    torch_image = torch_image.permute(0, 3, 1, 2)
    torch_image = torch_image.float()
    torch_image /= 255
    torch_image = torch_image.cuda()
    # fdf = torch_src.numpy()
    # fdf = np.squeeze(fdf, axis=0)
    # fdf = np.moveaxis(fdf, 0, -1)

    out = model(torch_image)
    start_time = round(time.monotonic() * 1000)
    out = model(torch_image)
    time_taken = round(time.monotonic() * 1000) - start_time
    print("model(image)", time_taken, "ms")  # (3ms no debug, no GPU->CPU)

    # weights_torch = A.load_weights(torch_weights)
    # model_torch = A.KitModel(torch_weights)
    #import torch

#import keras_to_torch_uem_mask

#torch.load("/home/thermalview/Desktop/ThermalView/tests/keras_torch/keras_to_torch_uem_mask.pt")
# weights_torch = keras_to_torch_uem_mask.load_weights("/home/thermalview/Desktop/ThermalView/tests/keras_torch/keras_to_torch_uem_mask.pt")
# torch_model = keras_to_torch_uem_mask.KitModel(weights_torch)
#torch_model = keras_to_torch_uem_mask.KitModel("/home/thermalview/Desktop/ThermalView/tests/keras_torch/keras_to_torch_uem_mask.pt")

import tensorflow as tf
from tensorflow.keras.models import load_model
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

# uem_mask_model = load_model("/home/thermalview/Desktop/ThermalView/mask_detection/uem_mask/ta946tl180_i_va0.9989517vl0.0014108_r01e0001_allw12vk2p1randd5v20-1-20-0-128l256l512l728l1024-128x128x3-sn.h5")
uem_mask_model = load_model("/home/thermalview/Desktop/ThermalView/mask_detection/uem_mask/ta947tl165_i_va0.9982527vl0.0022553_r01e0002_allw12vk2p1randd-5v20-sc-1-20-0-128l256l512l728l1024-128x128x3-sn.h5")
# in_frames_array = np.ones((1, 128, 128, 3), dtype=np.float32)
predictions = uem_mask_model.predict(test_img_128)
start_time = round(time.monotonic() * 1000)
predictions = uem_mask_model.predict(test_img_128)
time_taken = round(time.monotonic() * 1000) - start_time
print("uem_mask_model.predict", time_taken, "ms")  # (23 no debug, with GPU->CPU) 26-27-26-26-27 ms

test_img_160 = cv2.resize(test_img, (160, 160))
test_img_160 = np.expand_dims(test_img_160, 0)

from face_recognition_modules.deepface.DeepFace import build_model
facenet512_model = build_model("Facenet512")
# facenet512_model = Facenet512.loadModel("/home/thermalview/Desktop/ThermalView/face_recognition_modules/deepface/weights/facenet512_weights.h5")
# facenet512_model = facenet512_model()
# Facenet512.loadModel("/home/thermalview/Desktop/ThermalView/face_recognition_modules/deepface/weights/facenet512_weights.h5")
# in_frames_array = np.zeros((1, 160, 160, 3), dtype=np.uint8)
face_features1 = facenet512_model.predict(test_img_160)
start_time = round(time.monotonic() * 1000)
face_features1 = facenet512_model.predict(test_img_160)
time_taken = round(time.monotonic() * 1000) - start_time
print("facenet512_model.predict", time_taken, "ms")  # (26 no debug, with GPU->CPU) 37-33-32-32-34 ms

with torch.no_grad():
    timesler_model = TimeslerInceptionResnetV1(pretrained='vggface2').eval()
    timesler_model.cuda()
    # detected_face1 = np.expand_dims(detected_face, 0)
    torch_data = torch.from_numpy(test_img_160)
    # cuda0 = torch.device('cuda:0')
    torch_data = torch_data.cuda()  # .to(cuda0)
    torch_data = torch_data.permute(0, 3, 1, 2)
    # torch_data = torch.nn.functional.interpolate(torch_data, (160, 160))
    torch_data = torch_data.float()
    torch_data /= 255
    face_features2 = timesler_model(torch_data)
    start_time = round(time.monotonic() * 1000)
    face_features2 = timesler_model(torch_data)
    time_taken = round(time.monotonic() * 1000) - start_time
    print("timesler_model", time_taken, "ms")  # (26 no debug, with GPU->CPU) 37-33-32-32-34 ms
    face_features2 = face_features2.cpu().detach().numpy().squeeze()
    face_features2 = face_features2
