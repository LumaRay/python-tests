import os

import cv2
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Activation, MaxPooling3D, Dense, Dropout, Reshape, LSTM
import tensorflow as tf

from matplotlib import pyplot as plt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#cap = cv2.VideoCapture("sdfsf.jpg")
#cap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink max-buffers=1 drop=True")
#res, frame = cap.read()
#frame = cv2.resize(frame, (512, 512))

face_width = 64
face_height = 64

FACE_SEQUENCE_SIZE = 1#20
TRAIN_FACE_SEQUENCES_COUNT = 3000#500
VALID_FACE_SEQUENCES_COUNT = 300#50
TEST_FACE_SEQUENCES_COUNT = 300#50


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

#path_faces = "/home/thermalview/Desktop/ThermalView/alert/2020_11_25_30ms/faces/temperature"
#path_faces = "/home/thermalview/Desktop/ThermalView/alert/2020_11_25_1000ms/faces/temperature"
path_faces = "/home/thermalview/Desktop/ThermalView/alert/2020_11_19/faces/temperature"


path_color = path_faces + "/color/small"
path_depth = path_faces + "/depth/small"
path_thermal = path_faces + "/thermal/small"

x_train = np.zeros((TRAIN_FACE_SEQUENCES_COUNT, FACE_SEQUENCE_SIZE, face_height, face_width, 9), dtype=float)
y_train = np.zeros((TRAIN_FACE_SEQUENCES_COUNT, 1, 1), dtype=float)

x_val = np.zeros((VALID_FACE_SEQUENCES_COUNT, FACE_SEQUENCE_SIZE, face_height, face_width, 9), dtype=float)
y_val = np.zeros((VALID_FACE_SEQUENCES_COUNT, 1, 1), dtype=float)

x_test = np.zeros((TEST_FACE_SEQUENCES_COUNT, FACE_SEQUENCE_SIZE, face_height, face_width, 9), dtype=float)
y_test = np.zeros((TEST_FACE_SEQUENCES_COUNT, 1, 1), dtype=float)

list_faces = []
face_sequence_idx = 0
train_set_idx = 0
val_set_idx = 0
test_set_idx = 0
for file_color in os.listdir(path_color):
    if file_color.endswith(".jpg"):
        #if len(color_files) >= MAX_FACES:
        #    break
        cap = cv2.VideoCapture(path_color + "/" + file_color)
        res, frame_color = cap.read()
        frame_color = cv2.resize(frame_color, (face_width, face_height))
        file_parts_color = file_color.split("_")
        str_temp = ".".join(file_parts_color[-1].split(".")[:2])
        temp = float(str_temp)
        file_parts_color[-3] = "DEPTH"
        # file_depth = depth_files1[idx]
        file_depth = "_".join(file_parts_color)
        cap = cv2.VideoCapture(path_depth + "/" + file_depth)
        res, frame_depth = cap.read()
        frame_depth = cv2.resize(frame_depth, (face_width, face_height))
        file_parts_color[-3] = "THERMAL"
        # file_thermal = thermal_files1[idx]
        file_thermal = "_".join(file_parts_color)
        cap = cv2.VideoCapture(path_thermal + "/" + file_thermal)
        res, frame_thermal = cap.read()
        frame_thermal = cv2.resize(frame_thermal, (face_width, face_height))
        combined_frame = np.concatenate((frame_color, frame_depth, frame_thermal), axis=2)
        combined_frame = combined_frame.astype(np.float32) / 255
        # combined_frame = np.concatenate((combined_frame, np.full((face_height, face_width), temp)), axis=2)
        list_faces.append((combined_frame, temp))
        if len(list_faces) > FACE_SEQUENCE_SIZE:
            list_faces.pop(0)
            np_combined_frames = np.zeros((FACE_SEQUENCE_SIZE, face_height, face_width, 9), dtype=np.float32)
            np_temps = np.zeros((FACE_SEQUENCE_SIZE, 1), dtype=np.float32)
            for idx in range(FACE_SEQUENCE_SIZE):
                np_combined_frames[idx], np_temps[idx] = list_faces[idx]
            if face_sequence_idx < TRAIN_FACE_SEQUENCES_COUNT:
                x_train[train_set_idx] = np_combined_frames
                y_train[train_set_idx, 0] = np_temps[-1] / 100
                train_set_idx += 1
            elif face_sequence_idx < TRAIN_FACE_SEQUENCES_COUNT + VALID_FACE_SEQUENCES_COUNT:
                x_val[val_set_idx] = np_combined_frames
                y_val[val_set_idx, 0] = np_temps[-1] / 100
                val_set_idx += 1
            elif face_sequence_idx < TRAIN_FACE_SEQUENCES_COUNT + VALID_FACE_SEQUENCES_COUNT + TEST_FACE_SEQUENCES_COUNT:
                x_test[test_set_idx] = np_combined_frames
                y_test[test_set_idx, 0] = np_temps[-1] / 100
                test_set_idx += 1
            else:
                break
            face_sequence_idx += 1




history = model.fit(
    x_train,
    y_train,
    batch_size=10,#64,
    epochs=100,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
    verbose=1,
    shuffle=True,
    #validation_split=0.2,
)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

results = model.evaluate(x_test, y_test, batch_size=1)#128)
print("test loss, test acc:", results)

#predictions = model.predict(x_test[:3])
predictions = model.predict(x_test)
print("predictions shape:", predictions.shape)
