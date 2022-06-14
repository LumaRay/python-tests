import os

import cv2
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Activation, MaxPooling3D, Dense, Dropout, Reshape, LSTM, Flatten, BatchNormalization, Input
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
#571 vqlid samples
FACE_SEQUENCE_SIZE = 1#5#20
TRAIN_FACE_SEQUENCES_COUNT = 500#1#300#500
VALID_FACE_SEQUENCES_COUNT = 60#30#50
TEST_FACE_SEQUENCES_COUNT = 11#30#50

MAX_TEMP = 50.0
MIN_TEMP = 30.0

# use simple CNN structure
in_shape = (FACE_SEQUENCE_SIZE, face_width, face_height, 6)

input_1 = Input(shape=(10,))
x = Dense(64*64, activation="relu")(input_1)
x = Dense(4, activation="relu")(x)
x = Dense(1, activation="linear")(x)
model = Model(inputs, x)

'''model = Sequential()
model.add(Flatten(input_shape=in_shape))
model.add(Activation('relu'))
model.add(Dense(FACE_SEQUENCE_SIZE * face_width * face_height))
model.add(Activation('relu'))
model.add(Dense(FACE_SEQUENCE_SIZE * face_width * face_height))
model.add(Activation('relu'))
model.add(Dense(FACE_SEQUENCE_SIZE * face_width * face_height))
model.add(Activation('relu'))
model.add(Dense(FACE_SEQUENCE_SIZE * face_width * face_height))
model.add(Activation('relu'))
model.add(Dense(FACE_SEQUENCE_SIZE * face_width * face_height))
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))
#model.add(Dense(1, activation='linear'))
model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.SGD(), metrics=[keras.metrics.mae])
#model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam(), metrics=[keras.metrics.mae])
#opt = keras.optimizers.Adam(lr=1e-3, decay=1e-3 / 200)
#model.compile(loss=keras.losses.mean_absolute_percentage_error, optimizer=opt)'''

'''model = Sequential()
model.add(Flatten(input_shape=in_shape))
model.add(Activation('relu'))
model.add(Dense(64*64))
model.add(Activation('relu'))
#model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.1))
model.add(Dense(64*32))
model.add(Activation('relu'))
#model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.1))
model.add(Dense(64*32))
model.add(Activation('relu'))
#model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.1))
model.add(Dense(64*32))
model.add(Activation('relu'))
#model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.1))
model.add(Dense(64*32))
model.add(Activation('relu'))
#model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.1))
model.add(Dense(64*16))
model.add(Activation('relu'))
#model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.1))
model.add(Dense(64*8))
model.add(Activation('relu'))
#model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.1))
model.add(Dense(64*4))
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))
#model.add(Dense(1, activation='linear'))
model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.SGD(), metrics=[keras.metrics.mae])'''

'''model.add(ConvLSTM2D(32, kernel_size=(7, 7), padding='valid', return_sequences=True, input_shape=in_shape))
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
model.add(Reshape((FACE_SEQUENCE_SIZE, out_shape[2] * out_shape[3] * out_shape[4])))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
#model.add(Dense(1, activation='sigmoid'))
model.add(Dense(1, activation='linear'))
#model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=keras.optimizers.SGD(), metrics=[keras.metrics.Accuracy()])
#model.compile(loss=keras.losses.Poisson(), optimizer=keras.optimizers.SGD(), metrics=[keras.metrics.Accuracy()])
#model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.SGD(), metrics=[keras.metrics.mae])
model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam(), metrics=[keras.metrics.mae])
#opt = keras.optimizers.Adam(lr=1e-3, decay=1e-3 / 200)
#model.compile(loss=keras.losses.mean_absolute_percentage_error, optimizer=opt)'''


# model structure summary
print(model.summary())

#path_faces = "/home/thermalview/Desktop/ThermalView/alert/2020_11_25_30ms/faces/temperature"
#path_faces = "/home/thermalview/Desktop/ThermalView/alert/2020_11_25_1000ms/faces/temperature"
#path_faces = "/home/thermalview/Desktop/ThermalView/alert/2020_11_19/faces/temperature"
#path_faces = "/home/thermalview/Desktop/ThermalView/alert/2020_12_02/faces/temperature"
#path_faces = "/home/thermalview/Desktop/ThermalView/alert/2020_11_25_3000ms5max/faces/temperature"
#path_faces = "/home/thermalview/Desktop/ThermalView/alert/2020_12_07/faces/temperature"
path_faces = "/home/thermalview/Desktop/ThermalView/alert/2020_12_09/faces/temperature"

path_color = path_faces + "/color/small"
path_depth = path_faces + "/depth/data/small"
path_thermal = path_faces + "/thermal/data/parsed/small"
path_shape = path_faces + "/shape/data/small"

x_train = np.zeros((TRAIN_FACE_SEQUENCES_COUNT, FACE_SEQUENCE_SIZE, face_height, face_width, 6), dtype=np.float32)
y_train = np.zeros((TRAIN_FACE_SEQUENCES_COUNT, 1, 1), dtype=np.float32)

x_val = np.zeros((VALID_FACE_SEQUENCES_COUNT, FACE_SEQUENCE_SIZE, face_height, face_width, 6), dtype=np.float32)
y_val = np.zeros((VALID_FACE_SEQUENCES_COUNT, 1, 1), dtype=np.float32)

x_test = np.zeros((TEST_FACE_SEQUENCES_COUNT, FACE_SEQUENCE_SIZE, face_height, face_width, 6), dtype=np.float32)
y_test = np.zeros((TEST_FACE_SEQUENCES_COUNT, 1, 1), dtype=np.float32)

list_faces = []
face_sequence_idx = 0
train_set_idx = 0
val_set_idx = 0
test_set_idx = 0
list_dir = os.listdir(path_color)
if FACE_SEQUENCE_SIZE == 1:
    from random import shuffle
    shuffle(list_dir)
for file_color in list_dir:
    if file_color.endswith(".jpg"):
        #if len(color_files) >= MAX_FACES:
        #    break
        try:
            #cap = cv2.VideoCapture(path_color + "/" + file_color)
            #res, frame_color = cap.read()
            frame_color = cv2.imread(path_color + "/" + file_color)
            frame_color = cv2.resize(frame_color, (face_width, face_height))
            frame_color = (frame_color / 256).astype(np.float32)
            file_parts_color = file_color.split("_")
            str_temp = ".".join(file_parts_color[-1].split(".")[:2])
            temp = (float(str_temp) - MIN_TEMP) / (MAX_TEMP - MIN_TEMP)

            file_parts_color[-3] = "DEPTH_DATA"
            file_parts_color[-1] = ".".join(file_parts_color[-1].split(".")[:2]) + ".depth.npy"
            # file_depth = depth_files1[idx]
            file_depth = "_".join(file_parts_color)
            #cap = cv2.VideoCapture(path_depth + "/" + file_depth)
            #res, frame_depth = cap.read()
            frame_depth = np.load(path_depth + "/" + file_depth)
            frame_depth = cv2.resize(frame_depth, (face_width, face_height))
            frame_depth = np.expand_dims((frame_depth / 65536).astype(np.float32), axis=2)

            file_parts_color[-3] = "THERMAL_DATA_PARSED"
            file_parts_color[-1] = ".".join(file_parts_color[-1].split(".")[:2]) + ".thermal-parsed.npy"
            # file_thermal = thermal_files1[idx]
            file_thermal = "_".join(file_parts_color)
            #cap = cv2.VideoCapture(path_thermal + "/" + file_thermal)
            #res, frame_thermal = cap.read()
            frame_thermal = np.load(path_thermal + "/" + file_thermal)
            frame_thermal = (frame_thermal - MIN_TEMP) / (MAX_TEMP - MIN_TEMP)
            frame_thermal = cv2.resize(frame_thermal, (face_width, face_height))
            frame_thermal = np.expand_dims(frame_thermal, axis=2)

            file_parts_color[-3] = "SHAPE_DATA"
            file_parts_color[-1] = ".".join(file_parts_color[-1].split(".")[:2]) + ".shape.npy"
            file_shape = "_".join(file_parts_color)
            frame_shape = np.load(path_shape + "/" + file_shape)
            frame_shape = cv2.resize(frame_shape, (face_width, face_height))
            frame_shape = np.expand_dims(frame_shape, axis=2)

            combined_frame = np.concatenate((frame_color, frame_depth, frame_thermal, frame_shape), axis=2)
            #combined_frame = combined_frame.astype(np.float32) / 255
            # combined_frame = np.concatenate((combined_frame, np.full((face_height, face_width), temp)), axis=2)
        except:
            continue
        list_faces.append((combined_frame, temp))
        if len(list_faces) == FACE_SEQUENCE_SIZE:
            np_combined_frames = np.zeros((FACE_SEQUENCE_SIZE, face_height, face_width, 6), dtype=np.float32)
            np_temps = np.zeros((FACE_SEQUENCE_SIZE, 1), dtype=np.float32)
            for idx in range(FACE_SEQUENCE_SIZE):
                np_combined_frames[idx], np_temps[idx] = list_faces[idx]
            if face_sequence_idx < TRAIN_FACE_SEQUENCES_COUNT:
                x_train[train_set_idx] = np_combined_frames
                y_train[train_set_idx, 0] = np_temps[-1]
                train_set_idx += 1
            elif face_sequence_idx < TRAIN_FACE_SEQUENCES_COUNT + VALID_FACE_SEQUENCES_COUNT:
                x_val[val_set_idx] = np_combined_frames
                y_val[val_set_idx, 0] = np_temps[-1]
                val_set_idx += 1
            elif face_sequence_idx < TRAIN_FACE_SEQUENCES_COUNT + VALID_FACE_SEQUENCES_COUNT + TEST_FACE_SEQUENCES_COUNT:
                x_test[test_set_idx] = np_combined_frames
                y_test[test_set_idx, 0] = np_temps[-1]
                test_set_idx += 1
            else:
                break
            face_sequence_idx += 1
            list_faces.pop(0)


path_checkpoint = "model_checkpoint.h5"
#es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=15)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    x_train,
    y_train,
    batch_size=10,#5,#10,#64,
    epochs=100,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
    verbose=1,
    shuffle=True,
    callbacks=[es_callback, modelckpt_callback],
    #validation_split=0.2,
)

try:
    # summarize history for accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
except:
    pass

try:
    # summarize history for loss
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model mean_absolute_error')
    plt.ylabel('mean_absolute_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
except:
    pass

results = model.evaluate(x_test, y_test, batch_size=1)#128)
print("test loss, test acc:", results)

#predictions = model.predict(x_test[:3])
predictions = model.predict(x_test)
print("predictions shape:", predictions.shape)
print("prediction [0, 0]:", predictions[0, 0] * MAX_TEMP + MIN_TEMP, " should be:", y_test[0, 0, 0] * MAX_TEMP + MIN_TEMP)

input("Press Enter to continue...")