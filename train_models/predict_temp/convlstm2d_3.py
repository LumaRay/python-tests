import os

import cv2
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Activation, MaxPooling3D, Dense, Dropout, Reshape, LSTM
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#cap = cv2.VideoCapture("sdfsf.jpg")
#cap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink max-buffers=1 drop=True")
#res, frame = cap.read()
#frame = cv2.resize(frame, (512, 512))

face_width = 64
face_height = 64

MAX_FACES = 1000
FACE_SEQUENCE_SIZE = 20
TRAIN_FACE_SEQUENCES_COUNT = 50


# use simple CNN structure
in_shape = (FACE_SEQUENCE_SIZE, face_width, face_height, 9)
model = Sequential()
model.add(ConvLSTM2D(32, kernel_size=(7, 7), padding='valid', return_sequences=True, input_shape=in_shape))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
model.add(ConvLSTM2D(64, kernel_size=(5, 5), padding='valid', return_sequences=True))
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
model.add(ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True))
model.add(Activation('relu'))
model.add(ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True))
model.add(Activation('relu'))
model.add(ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True))
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
model.add(Dense(320))
model.add(Activation('relu'))
model.add(Dropout(0.5))

out_shape = model.output_shape
# print('====Model shape: ', out_shape)
model.add(Reshape((FACE_SEQUENCE_SIZE, out_shape[2] * out_shape[3] * out_shape[4])))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
#model.add(Dense(N_CLASSES, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))
#model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=keras.optimizers.SGD(), metrics=[keras.metrics.Accuracy()])
#model.compile(loss=keras.losses.Poisson(), optimizer=keras.optimizers.SGD(), metrics=[keras.metrics.Accuracy()])
#model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.SGD(), metrics=[keras.metrics.mae])
model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam(), metrics=[keras.metrics.mae])

# model structure summary
print(model.summary())


path_color = "/home/thermalview/Desktop/ThermalView/alert/2020_11_25_30ms/faces/temperature/color/small"
path_depth = "/home/thermalview/Desktop/ThermalView/alert/2020_11_25_30ms/faces/temperature/depth/small"
path_thermal = "/home/thermalview/Desktop/ThermalView/alert/2020_11_25_30ms/faces/temperature/thermal/small"


def loadFrames(color_files):
    combined_frames = np.zeros((len(color_files), face_height, face_width, 9), dtype=np.float32)
    temps = np.zeros((len(color_files), 1), dtype=np.float32)
    for idx in range(len(color_files)):
        file_color = color_files[idx]
        cap = cv2.VideoCapture(path_color + "/" + file_color)
        res, frame_color = cap.read()
        frame_color = cv2.resize(frame_color, (face_width, face_height))
        file_parts_color = file_color.split("_")
        str_temp = ".".join(file_parts_color[11].split(".")[:2])
        temp = float(str_temp)
        file_parts_color[9] = "DEPTH"
        #file_depth = depth_files1[idx]
        file_depth = "_".join(file_parts_color)
        cap = cv2.VideoCapture(path_depth + "/" + file_depth)
        res, frame_depth = cap.read()
        frame_depth = cv2.resize(frame_depth, (face_width, face_height))
        file_parts_color[9] = "THERMAL"
        #file_thermal = thermal_files1[idx]
        file_thermal = "_".join(file_parts_color)
        cap = cv2.VideoCapture(path_thermal + "/" + file_thermal)
        res, frame_thermal = cap.read()
        frame_thermal = cv2.resize(frame_thermal, (face_width, face_height))
        combined_frame = np.concatenate((frame_color, frame_depth, frame_thermal), axis=2)
        combined_frame = combined_frame.astype(np.float32) / 255
        #combined_frame = np.concatenate((combined_frame, np.full((face_height, face_width), temp)), axis=2)
        combined_frames[idx] = combined_frame
        temps[idx] = temp
    return combined_frames, temps

color_files = []
for file in os.listdir(path_color):
    if file.endswith(".jpg"):
        #color_files.append(path_color + "/" + file)
        color_files.append(file)
        if len(color_files) >= MAX_FACES:
            break


x_train = np.zeros((TRAIN_FACE_SEQUENCES_COUNT, FACE_SEQUENCE_SIZE, face_height, face_width, 9), dtype=float)
y_train = np.zeros((TRAIN_FACE_SEQUENCES_COUNT, 1, 1), dtype=float)


color_faces_list = []
train_set_idx = 0
for color_file in color_files:
    color_faces_list.append(color_file)
    if len(color_faces_list) > FACE_SEQUENCE_SIZE:
        color_faces_list.pop(0)
        combined_frames, temps = loadFrames(color_faces_list)
        if train_set_idx < TRAIN_FACE_SEQUENCES_COUNT:
            x_train[train_set_idx] = combined_frames
            y_train[train_set_idx, 0] = temps[-1] / 100
        else:
            break
        train_set_idx += 1

history = model.fit(
    x_train,
    y_train,
    batch_size=10,#64,
    epochs=2000,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    #validation_data=(x_val, y_val),
    verbose=1,
    validation_split=0.2,
)
