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

face_width = 128
face_height = 128

SequenceLength = 6
N_CLASSES = 1

# use simple CNN structure
in_shape = (SequenceLength, face_width, face_height, 9)
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
model.add(Reshape((SequenceLength, out_shape[2] * out_shape[3] * out_shape[4])))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
#model.add(Dense(N_CLASSES, activation='softmax'))
model.add(Dense(N_CLASSES, activation='sigmoid'))
#model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=keras.optimizers.SGD(), metrics=[keras.metrics.Accuracy()])
model.compile(loss=keras.losses.Poisson(), optimizer=keras.optimizers.SGD(), metrics=[keras.metrics.Accuracy()])

# model structure summary
print(model.summary())


path_color = "/home/thermalview/Desktop/ThermalView/alert/2020_11_25_30ms/faces/temperature/color/small"

color_files1 = []
color_files1.append("2020_11_25__11_45_56_268695_TEMP_COLOR_SMALL_40.164641606950056.jpg")
color_files1.append("2020_11_25__11_45_56_349765_TEMP_COLOR_SMALL_40.261714114595364.jpg")
color_files1.append("2020_11_25__11_45_56_437591_TEMP_COLOR_SMALL_40.947120340029905.jpg")
color_files1.append("2020_11_25__11_45_56_507861_TEMP_COLOR_SMALL_40.947120340029905.jpg")
color_files1.append("2020_11_25__11_45_56_585688_TEMP_COLOR_SMALL_40.947120340029905.jpg")
color_files1.append("2020_11_25__11_45_56_667452_TEMP_COLOR_SMALL_40.947120340029905.jpg")

color_files2 = []
color_files2.append("2020_11_25__11_45_56_744812_TEMP_COLOR_SMALL_39.91617358897305.jpg")
color_files2.append("2020_11_25__11_45_56_833975_TEMP_COLOR_SMALL_40.0197753940865.jpg")
color_files2.append("2020_11_25__11_45_56_911094_TEMP_COLOR_SMALL_40.0197753940865.jpg")
color_files2.append("2020_11_25__11_45_56_991368_TEMP_COLOR_SMALL_40.0197753940865.jpg")
color_files2.append("2020_11_25__11_45_57_067697_TEMP_COLOR_SMALL_40.0197753940865.jpg")
color_files2.append("2020_11_25__11_45_57_146026_TEMP_COLOR_SMALL_40.0197753940865.jpg")

color_files3 = []
color_files3.append("2020_11_25__11_46_11_584885_TEMP_COLOR_SMALL_36.91028995644343.jpg")
color_files3.append("2020_11_25__11_46_11_664706_TEMP_COLOR_SMALL_36.91028995644343.jpg")
color_files3.append("2020_11_25__11_46_11_747992_TEMP_COLOR_SMALL_36.91028995644343.jpg")
color_files3.append("2020_11_25__11_46_11_830733_TEMP_COLOR_SMALL_36.91028995644343.jpg")
color_files3.append("2020_11_25__11_46_11_904311_TEMP_COLOR_SMALL_36.91028995644343.jpg")
color_files3.append("2020_11_25__11_46_12_071972_TEMP_COLOR_SMALL_36.91028995644343.jpg")

path_depth = "/home/thermalview/Desktop/ThermalView/alert/2020_11_25_30ms/faces/temperature/depth/small"

path_thermal = "/home/thermalview/Desktop/ThermalView/alert/2020_11_25_30ms/faces/temperature/thermal/small"



def loadFrames(color_files):
    combined_frames = np.zeros((len(color_files), face_height, face_width, 9), dtype=np.float32)
    temps = np.zeros((len(color_files), 1), dtype=np.float32)
    for idx in range(len(color_files)):
        file_color = color_files1[idx]
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

combined_frames1, temps1 = loadFrames(color_files1)
combined_frames2, temps2 = loadFrames(color_files2)
combined_frames3, temps3 = loadFrames(color_files3)

x_train = np.zeros((1, len(combined_frames1), face_height, face_width, 9), dtype=float)
x_train[0] = combined_frames1

#y_train = np.zeros((1, len(temps1), 1), dtype=float)
#y_train[0] = temps1 / 100
y_train = np.zeros((1, 1, 1), dtype=float)
y_train[0, 0] = temps1[-1] / 100

x_val = np.zeros((1, len(combined_frames2), face_height, face_width, 9), dtype=float)
x_val[0] = combined_frames2

#y_val = np.zeros((1, len(temps2), 1), dtype=float)
#y_val[0] = temps2 / 100
y_val = np.zeros((1, 1, 1), dtype=float)
y_val[0, 0] = temps2[-1] / 100

x_test = np.zeros((1, len(combined_frames3), face_height, face_width, 9), dtype=float)
x_test[0] = combined_frames3

#y_test = np.zeros((1, len(temps3), 1), dtype=float)
#y_test[0] = temps3 / 100
y_test = np.zeros((1, 1, 1), dtype=float)
y_test[0, 0] = temps3[-1] / 100

#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#y_train = y_train.astype("float32")
#y_test = y_test.astype("float32")


history = model.fit(
    x_train,
    y_train,
    batch_size=1,#64,
    epochs=2000,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)
