# https://keras.io/examples/vision/mnist_convnet/
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import cv2
import os
import pathlib
pathToScriptFolder = str(pathlib.Path().absolute())

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

buy_width, buy_height = 170, 310
sell_width, sell_height = 170, 320
hist_width, hist_height = 170, 710
chart_width, chart_height = 600, 300
# res_height, res_width = 248, 136
# res_width, res_height = 480, 240
res_width, res_height = 576, 288

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

MAX_SAMPLES = 5000
HISTORICAL_FRAMES = 1
# HISTORICAL_FRAMES = 4
COLOR_CHANNELS = 3
VALID_RATIO = 25
TEST_RATIO = 10

MODEL_TYPE_DEFAULT = "default"
MODEL_TYPE_REVERSED = "reversed"
MODEL_TYPE_REVERSED2 = "reversed2"
MODEL_TYPE_REVERSED3 = "reversed3"  # +++
MODEL_TYPE_REVERSED3_DENSE = "reversed3_dense"  # +++
MODEL_TYPE_REVERSED3_DENSE2 = "reversed3_dense2"  # +
MODEL_TYPE_REVERSED4 = "reversed4"
MODEL_TYPE_LSTM = "lstm"
MODEL_TYPE_LSTM_REVERSED = "lstm_reversed"
MODEL_TYPE_LSTM_REVERSED2 = "lstm_reversed2"   # +++
MODEL_TYPE_LSTM_REVERSED2_DENSE = "lstm_reversed2_dense"  # chart+++
MODEL_TYPE = MODEL_TYPE_LSTM_REVERSED2_DENSE

DATASET_TYPE_MNIST = "mnist"
DATASET_TYPE_CHART = "chart"
DATASET_TYPE = DATASET_TYPE_CHART

# DATASET_GRAY = False
DATASET_GRAY = True
if DATASET_GRAY:
    COLOR_CHANNELS = 1

if DATASET_TYPE == DATASET_TYPE_MNIST:
    NUM_EPOCHS = 10
    # BATCH_SIZE = 1
    BATCH_SIZE = 128

if DATASET_TYPE == DATASET_TYPE_CHART:
    NUM_EPOCHS = 10
    BATCH_SIZE = 1

def prepareMNIST():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    # x_train = x_train.astype("float32") / 255
    # x_train = x_train.astype("float32")
    # x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")


    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test

def prepareCustom():
    IMPORT_FOLDER = pathToScriptFolder + f"/frames_src/simple-line-10s-span10min-es2/2021-05-11-12-27-38-361400/"

    lst_buy = []
    lst_sell = []
    lst_hist = []
    lst_chart = []

    files = sorted(os.listdir(IMPORT_FOLDER))
    for fidx, file in enumerate(files):
        if len(lst_buy) >= MAX_SAMPLES:
            break
        if file.endswith(".jpg"):
            file_name, file_ext = file.split('.')
            # print("Parsing file " + str(fidx) + " of " + str(len(files)))
            frame_timestamp, type = file_name.split('_')
            frame_color = cv2.imread(IMPORT_FOLDER + file)
            if DATASET_GRAY:
                frame_color = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
            # frame_color = cv2.resize(frame_color, (face_width, face_height))
            frame_color = cv2.normalize(frame_color, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            if DATASET_GRAY:
                frame_color = np.expand_dims(frame_color, -1)
            # frame_color = (frame_color / 256).astype(np.float32)
            if type == "buy":
                lst_buy.append(frame_color)
            if type == "sell":
                lst_sell.append(frame_color)
            if type == "hist":
                lst_hist.append(frame_color)
            if type == "chart":
                lst_chart.append(frame_color)
                # cv2.imshow(file_name, frame_color)
                # cv2.waitKey(0)

    total_samples = len(lst_buy) - HISTORICAL_FRAMES - 1

    x_samples_buy = np.zeros((total_samples, HISTORICAL_FRAMES, buy_height, buy_width, COLOR_CHANNELS), dtype=np.float32)
    x_samples_sell = np.zeros((total_samples, HISTORICAL_FRAMES, sell_height, sell_width, COLOR_CHANNELS), dtype=np.float32)
    x_samples_hist = np.zeros((total_samples, HISTORICAL_FRAMES, hist_height, hist_width, COLOR_CHANNELS), dtype=np.float32)
    x_samples_chart = np.zeros((total_samples, HISTORICAL_FRAMES, chart_height, chart_width, COLOR_CHANNELS), dtype=np.float32)
    y_samples = np.zeros((total_samples, res_height, res_width, COLOR_CHANNELS), dtype=np.float32)

    for idx_start in range(total_samples):
        x_samples_buy[idx_start] = np.asarray([img_buy for img_buy in lst_buy[idx_start:idx_start + HISTORICAL_FRAMES]])
        x_samples_sell[idx_start] = np.asarray([img_sell for img_sell in lst_sell[idx_start:idx_start + HISTORICAL_FRAMES]])
        x_samples_hist[idx_start] = np.asarray([img_hist for img_hist in lst_hist[idx_start:idx_start + HISTORICAL_FRAMES]])
        x_samples_chart[idx_start] = np.asarray([img_chart for img_chart in lst_chart[idx_start:idx_start + HISTORICAL_FRAMES]])
        # y_samples[idx_start] = cv2.resize(lst_buy[idx_start + HISTORICAL_FRAMES + 1], (res_width, res_height), cv2.INTER_CUBIC)
        tmp = cv2.resize(lst_chart[idx_start + HISTORICAL_FRAMES], (res_width, res_height), cv2.INTER_CUBIC)
        if DATASET_GRAY:
            tmp = np.expand_dims(tmp, -1)
        y_samples[idx_start] = tmp

    valid_samples_count = int(len(x_samples_buy) * VALID_RATIO / 100)
    test_samples_count = int(len(x_samples_buy) * TEST_RATIO / 100)
    train_samples_count = len(x_samples_buy) - valid_samples_count - test_samples_count

    x_train_buy = x_samples_buy[:train_samples_count]
    x_train_sell = x_samples_sell[:train_samples_count]
    x_train_hist = x_samples_hist[:train_samples_count]
    x_train_chart = x_samples_chart[:train_samples_count]
    y_train = y_samples[:train_samples_count]
    x_valid_buy = x_samples_buy[train_samples_count:-test_samples_count]
    x_valid_sell = x_samples_sell[train_samples_count:-test_samples_count]
    x_valid_hist = x_samples_hist[train_samples_count:-test_samples_count]
    x_valid_chart = x_samples_chart[train_samples_count:-test_samples_count]
    y_valid = y_samples[train_samples_count:-test_samples_count]
    x_test_buy = x_samples_buy[-test_samples_count:]
    x_test_sell = x_samples_sell[-test_samples_count:]
    x_test_hist = x_samples_hist[-test_samples_count:]
    x_test_chart = x_samples_chart[-test_samples_count:]
    y_test = y_samples[-test_samples_count:]

    return x_train_chart, y_train, x_test_chart, y_test

if DATASET_TYPE == DATASET_TYPE_MNIST:
    x_train, y_train, x_test, y_test = prepareMNIST()

if DATASET_TYPE == DATASET_TYPE_CHART:
    x_train, y_train, x_test, y_test = prepareCustom()

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
        # layers.experimental.preprocessing.RandomWidth(0.1, "cubic"),
        # layers.experimental.preprocessing.RandomContrast(0.2)
    ]
)

def make_model_default():
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = data_augmentation(x)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, kernel_size=(3, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes)(x)
    x = layers.Activation("softmax")(x)
    outputs = x
    model = keras.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def make_model_reversed():
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = data_augmentation(x)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, kernel_size=(3, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    # x = layers.Dense(num_classes)(x)
    x = layers.Dense(25)(x)
    x = layers.Activation("softmax")(x)
    x = layers.Reshape((5, 5, 1))(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(192, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(96, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(48, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(1, (1, 1), padding='same')(x)
    x = layers.Activation("sigmoid")(x)
    outputs = x
    model = keras.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def make_model_reversed2():
    inputs = keras.Input(shape=input_shape)
    x = inputs
    # x = data_augmentation(x)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(48, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(96, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(192, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(192, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(192, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(96, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(48, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(1, (1, 1), padding='same')(x)
    x = layers.Activation("sigmoid")(x)
    outputs = x
    model = keras.Model(inputs, outputs)
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    lr = 0.0008
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')
    return model

def make_model_reversed3():
    inputs = keras.Input(shape=input_shape)
    x = inputs
    # x = data_augmentation(x)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(48, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(96, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = layers.Conv2D(192, (3, 3), padding='same')(x)
    # x = layers.Activation("relu")(x)
    # x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = layers.Conv2D(192, (5, 5), padding='same')(x)
    # x = layers.Activation("relu")(x)
    # x = layers.UpSampling2D(size=(2, 2))(x)
    # x = layers.Conv2D(192, (3, 3), padding='same')(x)
    # x = layers.Activation("relu")(x)
    # x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(96, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(48, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(24, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(1, (1, 1), padding='same')(x)
    x = layers.Activation("sigmoid")(x)
    outputs = x
    model = keras.Model(inputs, outputs)
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    lr = 0.0008
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')
    return model

def make_model_reversed3_dense():
    inputs = keras.Input(shape=input_shape)
    x = inputs
    # x = data_augmentation(x)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(48, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(96, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = layers.Conv2D(192, (3, 3), padding='same')(x)
    # x = layers.Activation("relu")(x)
    # x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = layers.Conv2D(192, (5, 5), padding='same')(x)
    # x = layers.Activation("relu")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(x.shape[1])(x)
    x = layers.Activation("sigmoid")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Reshape((7, 7, 96))(x)

    # x = layers.UpSampling2D(size=(2, 2))(x)
    # x = layers.Conv2D(192, (3, 3), padding='same')(x)
    # x = layers.Activation("relu")(x)
    # x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(96, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(48, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(24, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(1, (1, 1), padding='same')(x)
    x = layers.Activation("sigmoid")(x)
    outputs = x
    model = keras.Model(inputs, outputs)
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    lr = 0.0008
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')
    return model

def make_model_reversed3_dense2():
    inputs = keras.Input(shape=input_shape)
    x = inputs
    # x = data_augmentation(x)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(48, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(96, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = layers.Conv2D(192, (3, 3), padding='same')(x)
    # x = layers.Activation("relu")(x)
    # x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = layers.Conv2D(192, (5, 5), padding='same')(x)
    # x = layers.Activation("relu")(x)

    x = layers.Flatten()(x)
    x_shape = x.shape[1]
    x = layers.Dense(x_shape)(x)
    x = layers.Activation("sigmoid")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(x_shape * 4)(x)
    x = layers.Activation("sigmoid")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(x_shape)(x)
    x = layers.Activation("sigmoid")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Reshape((7, 7, 96))(x)

    # x = layers.UpSampling2D(size=(2, 2))(x)
    # x = layers.Conv2D(192, (3, 3), padding='same')(x)
    # x = layers.Activation("relu")(x)
    # x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(96, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(48, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(24, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(1, (1, 1), padding='same')(x)
    x = layers.Activation("sigmoid")(x)
    outputs = x
    model = keras.Model(inputs, outputs)
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    lr = 0.0008
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')
    return model

def make_model_reversed4():
    inputs = keras.Input(shape=input_shape)
    x = inputs
    # x = data_augmentation(x)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(48, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(96, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = layers.Conv2D(192, (3, 3), padding='same')(x)
    # x = layers.Activation("relu")(x)
    # x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = layers.Conv2D(192, (5, 5), padding='same')(x)
    # x = layers.Activation("relu")(x)
    # x = layers.UpSampling2D(size=(2, 2))(x)
    # x = layers.Conv2D(192, (3, 3), padding='same')(x)
    # x = layers.Activation("relu")(x)
    # x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(96, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(48, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(24, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(1, (1, 1), padding='same')(x)
    x = layers.Activation("sigmoid")(x)
    outputs = x
    model = keras.Model(inputs, outputs)
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    lr = 0.0008
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')
    return model

def make_model_lstm():
    inputs = keras.Input(shape=input_shape)
    x = inputs
    if MODEL_TYPE == MODEL_TYPE_DEFAULT:
        x = data_augmentation(x)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    # x = layers.Conv2D(32, kernel_size=(3, 3))(x)
    x = layers.ConvLSTM2D(32, kernel_size=(3, 3), padding='valid', return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    # x = layers.Conv2D(64, kernel_size=(3, 3))(x)
    x = layers.ConvLSTM2D(64, kernel_size=(3, 3), padding='valid', return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes)(x)
    x = layers.Activation("softmax")(x)
    outputs = x
    model = keras.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def make_model_lstm_reversed():
    inputs = keras.Input(shape=input_shape)
    x = inputs
    if MODEL_TYPE == MODEL_TYPE_DEFAULT:
        x = data_augmentation(x)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    # x = layers.Conv2D(32, kernel_size=(3, 3))(x)
    x = layers.ConvLSTM2D(32, kernel_size=(3, 3), padding='valid', return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    # x = layers.Conv2D(64, kernel_size=(3, 3))(x)
    x = layers.ConvLSTM2D(64, kernel_size=(3, 3), padding='valid', return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    # x = layers.Dense(num_classes)(x)
    x = layers.Dense(25)(x)
    x = layers.Activation("softmax")(x)
    x = layers.Reshape((5, 5, 1))(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(192, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(96, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(48, (5, 5), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(1, (1, 1), padding='same')(x)
    x = layers.Activation("sigmoid")(x)
    outputs = x
    model = keras.Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def make_model_lstm_reversed2():
    inputs = keras.Input(shape=input_shape)
    x = inputs
    # x = data_augmentation(x)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    # x = layers.Conv2D(32, kernel_size=(3, 3))(x)
    x = layers.ConvLSTM2D(48, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    # x = layers.Conv2D(64, kernel_size=(3, 3))(x)
    # x = layers.ConvLSTM2D(96, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = layers.ConvLSTM2D(96, kernel_size=(3, 3), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = layers.Flatten()(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(num_classes)(x)
    # x = layers.Dense(25)(x)
    # x = layers.Activation("softmax")(x)
    # x = layers.Reshape((7, 7, 96))(x)
    # x = layers.UpSampling2D(size=(2, 2))(x)
    # x = layers.Conv2D(192, (3, 3), padding='same')(x)
    # x = layers.Activation("relu")(x)
    # x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(96, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(48, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(24, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(1, (1, 1), padding='same')(x)
    x = layers.Activation("sigmoid")(x)
    outputs = x
    model = keras.Model(inputs, outputs)
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    lr = 0.0008
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')
    return model

def make_model_lstm_reversed2_dense():
    inputs = keras.Input(shape=input_shape)
    x = inputs
    # x = data_augmentation(x)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    # x = layers.Conv2D(32, kernel_size=(3, 3))(x)
    x = layers.ConvLSTM2D(48, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    # x = layers.Conv2D(64, kernel_size=(3, 3))(x)
    # x = layers.ConvLSTM2D(96, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = layers.ConvLSTM2D(96, kernel_size=(3, 3), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = layers.Flatten()(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(num_classes)(x)
    # x = layers.Dense(25)(x)
    # x = layers.Activation("softmax")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(x.shape[1])(x)
    x = layers.Activation("sigmoid")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Reshape((7, 7, 96))(x)

    # x = layers.Reshape((7, 7, 96))(x)
    # x = layers.UpSampling2D(size=(2, 2))(x)
    # x = layers.Conv2D(192, (3, 3), padding='same')(x)
    # x = layers.Activation("relu")(x)
    # x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(96, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(48, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(24, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(1, (1, 1), padding='same')(x)
    x = layers.Activation("sigmoid")(x)
    outputs = x
    model = keras.Model(inputs, outputs)
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    lr = 0.0008
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')
    return model

def make_model_lstm_reversed2_chart():
    inputs = keras.Input(shape=input_shape)
    x = inputs
    # x = data_augmentation(x)
    # x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    # x = layers.Conv2D(32, kernel_size=(3, 3))(x)
    x = layers.ConvLSTM2D(48, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    # x = layers.Conv2D(64, kernel_size=(3, 3))(x)
    # x = layers.ConvLSTM2D(96, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = layers.ConvLSTM2D(96, kernel_size=(3, 3), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = layers.Flatten()(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(num_classes)(x)
    # x = layers.Dense(25)(x)
    # x = layers.Activation("softmax")(x)
    # x = layers.Reshape((7, 7, 96))(x)
    # x = layers.UpSampling2D(size=(2, 2))(x)
    # x = layers.Conv2D(192, (3, 3), padding='same')(x)
    # x = layers.Activation("relu")(x)
    # x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(96, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(48, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(24, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(3, (1, 1), padding='same')(x)
    x = layers.Activation("sigmoid")(x)
    outputs = x
    model = keras.Model(inputs, outputs)
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    lr = 0.0008
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')
    return model

def make_model_lstm_reversed2_chart_dense():
    inputs = keras.Input(shape=input_shape)
    x = inputs
    # x = data_augmentation(x)
    # x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    # x = layers.Conv2D(32, kernel_size=(3, 3))(x)
    x = layers.ConvLSTM2D(48, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("tanh")(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    # x = layers.Conv2D(64, kernel_size=(3, 3))(x)
    # x = layers.ConvLSTM2D(96, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = layers.ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("tanh")(x)
    # x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    # x = layers.ConvLSTM2D(96, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = layers.ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("tanh")(x)
    # x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    # x = layers.ConvLSTM2D(384, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = layers.ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("tanh")(x)
    # x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    # x = layers.ConvLSTM2D(768, kernel_size=(3, 3), padding='same')(x)
    x = layers.ConvLSTM2D(64, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("tanh")(x)
    # x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = layers.Flatten()(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(num_classes)(x)
    # x = layers.Dense(25)(x)
    # x = layers.Activation("softmax")(x)

    x_shape = x.shape

    x = layers.Flatten()(x)
    x_shape1 = x.shape[1]
    # x = layers.Dense(2000)(x)
    # x = layers.Activation("sigmoid")(x)
    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(x_shape1)(x)
    x = layers.Activation("sigmoid")(x)
    x = layers.Dropout(0.5)(x)
    # x = layers.Reshape((18, 37, 384))(x)
    # x = layers.Reshape((9, 18, 768))(x)
    # x = layers.Reshape((9, 18, 64))(x)
    x = layers.Reshape((x_shape[1], x_shape[2], x_shape[3]))(x)

    # x = layers.Reshape((7, 7, 96))(x)
    # x = layers.UpSampling2D(size=(2, 2))(x)
    # x = layers.Conv2D(768, (3, 3), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    # x = layers.Conv2D(384, (3, 3), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    # x = layers.Conv2D(192, (3, 3), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    # x = layers.Conv2D(96, (3, 3), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(48, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(24, (3, 3), padding='same')(x)
    x = layers.Activation("relu")(x)
    if DATASET_GRAY:
        x = layers.Conv2D(1, (1, 1), padding='same')(x)
    else:
        x = layers.Conv2D(3, (1, 1), padding='same')(x)
    x = layers.Activation("sigmoid")(x)
    outputs = x
    model = keras.Model(inputs, outputs)
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    lr = 0.0008
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')
    return model

if MODEL_TYPE == MODEL_TYPE_DEFAULT:
    model = make_model_default()
elif MODEL_TYPE == MODEL_TYPE_REVERSED:
    model = make_model_reversed()
    y_train = np.zeros((x_train.shape[0], 40, 40, 1), np.uint8)
    for idx_y_sample in range(len(x_train)):
        y_train[idx_y_sample] = np.expand_dims(cv2.resize(x_train[idx_y_sample], (40, 40), cv2.INTER_CUBIC), -1)
    y_test = np.zeros((x_test.shape[0], 40, 40, 1), np.uint8)
    for idx_y_sample in range(len(x_test)):
        y_test[idx_y_sample] = np.expand_dims(cv2.resize(x_test[idx_y_sample], (40, 40), cv2.INTER_CUBIC), -1)
elif MODEL_TYPE == MODEL_TYPE_REVERSED2:
    model = make_model_reversed2()
    y_train = np.zeros((x_train.shape[0], 24, 24, 1), np.uint8)
    for idx_y_sample in range(len(x_train)):
        y_train[idx_y_sample] = np.expand_dims(cv2.resize(x_train[idx_y_sample], (24, 24), cv2.INTER_CUBIC), -1)
    y_test = np.zeros((x_test.shape[0], 24, 24, 1), np.uint8)
    for idx_y_sample in range(len(x_test)):
        y_test[idx_y_sample] = np.expand_dims(cv2.resize(x_test[idx_y_sample], (24, 24), cv2.INTER_CUBIC), -1)
elif MODEL_TYPE == MODEL_TYPE_REVERSED3:
    model = make_model_reversed3()
    y_train = np.zeros((x_train.shape[0], 28, 28, 1), np.uint8)
    for idx_y_sample in range(len(x_train)):
        y_train[idx_y_sample] = np.expand_dims(cv2.resize(x_train[idx_y_sample], (28, 28), cv2.INTER_CUBIC), -1)
    y_test = np.zeros((x_test.shape[0], 28, 28, 1), np.uint8)
    for idx_y_sample in range(len(x_test)):
        y_test[idx_y_sample] = np.expand_dims(cv2.resize(x_test[idx_y_sample], (28, 28), cv2.INTER_CUBIC), -1)
    '''y_train = x_train
    y_test = x_test'''
elif MODEL_TYPE == MODEL_TYPE_REVERSED3_DENSE:
    model = make_model_reversed3_dense()
    y_train = np.zeros((x_train.shape[0], 28, 28, 1), np.uint8)
    for idx_y_sample in range(len(x_train)):
        y_train[idx_y_sample] = np.expand_dims(cv2.resize(x_train[idx_y_sample], (28, 28), cv2.INTER_CUBIC), -1)
    y_test = np.zeros((x_test.shape[0], 28, 28, 1), np.uint8)
    for idx_y_sample in range(len(x_test)):
        y_test[idx_y_sample] = np.expand_dims(cv2.resize(x_test[idx_y_sample], (28, 28), cv2.INTER_CUBIC), -1)
    '''y_train = x_train
    y_test = x_test'''
elif MODEL_TYPE == MODEL_TYPE_REVERSED3_DENSE2:
    model = make_model_reversed3_dense2()
    y_train = np.zeros((x_train.shape[0], 28, 28, 1), np.uint8)
    for idx_y_sample in range(len(x_train)):
        y_train[idx_y_sample] = np.expand_dims(cv2.resize(x_train[idx_y_sample], (28, 28), cv2.INTER_CUBIC), -1)
    y_test = np.zeros((x_test.shape[0], 28, 28, 1), np.uint8)
    for idx_y_sample in range(len(x_test)):
        y_test[idx_y_sample] = np.expand_dims(cv2.resize(x_test[idx_y_sample], (28, 28), cv2.INTER_CUBIC), -1)
elif MODEL_TYPE == MODEL_TYPE_REVERSED4:
    model = make_model_reversed4()
    y_train = x_train
    y_test = x_test
elif MODEL_TYPE == MODEL_TYPE_LSTM:
    input_shape = (1, 28, 28, 1)
    model = make_model_lstm()
    x_train = np.expand_dims(x_train, 1)
elif MODEL_TYPE == MODEL_TYPE_LSTM_REVERSED:
    input_shape = (1, 28, 28, 1)
    model = make_model_lstm_reversed()
    x_train = np.expand_dims(x_train, 1)
    y_train = np.zeros((x_train.shape[0], 40, 40, 1), np.float32)
    for idx_y_sample in range(len(x_train)):
        y_train[idx_y_sample] = np.expand_dims(cv2.resize(x_train[idx_y_sample][0], (40, 40), cv2.INTER_CUBIC), -1).astype("float32") / 255
    x_test = np.expand_dims(x_test, 1)
    y_test = np.zeros((x_test.shape[0], 40, 40, 1), np.float32)
    for idx_y_sample in range(len(x_test)):
        y_test[idx_y_sample] = np.expand_dims(cv2.resize(x_test[idx_y_sample][0], (40, 40), cv2.INTER_CUBIC), -1).astype("float32") / 255
elif MODEL_TYPE == MODEL_TYPE_LSTM_REVERSED2:
    if DATASET_TYPE == DATASET_TYPE_MNIST:
        input_shape = (1, 28, 28, 1)
        model = make_model_lstm_reversed2()
        x_train = np.expand_dims(x_train, 1)
        y_train = np.zeros((x_train.shape[0], 28, 28, 1), np.uint8)
        for idx_y_sample in range(len(x_train)):
            y_train[idx_y_sample] = np.expand_dims(cv2.resize(x_train[idx_y_sample][0], (28, 28), cv2.INTER_CUBIC), -1)
        x_test = np.expand_dims(x_test, 1)
        y_test = np.zeros((x_test.shape[0], 28, 28, 1), np.uint8)
        for idx_y_sample in range(len(x_test)):
            y_test[idx_y_sample] = np.expand_dims(cv2.resize(x_test[idx_y_sample][0], (28, 28), cv2.INTER_CUBIC), -1)
        '''y_train = x_train
        y_test = x_test'''
    if DATASET_TYPE == DATASET_TYPE_CHART:
        input_shape = (1, 300, 600, 3)
        model = make_model_lstm_reversed2_chart()
        y_train = np.zeros((x_train.shape[0], 300, 600, 3), np.uint8)
        for idx_y_sample in range(len(x_train)):
            y_train[idx_y_sample] = cv2.resize(x_train[idx_y_sample][0], (600, 300), cv2.INTER_CUBIC)
        y_test = np.zeros((x_test.shape[0], 300, 600, 3), np.uint8)
        for idx_y_sample in range(len(x_test)):
            y_test[idx_y_sample] = cv2.resize(x_test[idx_y_sample][0], (600, 300), cv2.INTER_CUBIC)

elif MODEL_TYPE == MODEL_TYPE_LSTM_REVERSED2_DENSE:
    if DATASET_TYPE == DATASET_TYPE_MNIST:
        input_shape = (None, 28, 28, 1)
        model = make_model_lstm_reversed2_dense()
        x_train = np.expand_dims(x_train, 1)
        y_train = np.zeros((x_train.shape[0], 28, 28, 1), np.uint8)
        for idx_y_sample in range(len(x_train)):
            y_train[idx_y_sample] = np.expand_dims(cv2.resize(x_train[idx_y_sample][0], (28, 28), cv2.INTER_CUBIC), -1)
        x_test = np.expand_dims(x_test, 1)
        y_test = np.zeros((x_test.shape[0], 28, 28, 1), np.uint8)
        for idx_y_sample in range(len(x_test)):
            y_test[idx_y_sample] = np.expand_dims(cv2.resize(x_test[idx_y_sample][0], (28, 28), cv2.INTER_CUBIC), -1)
    if DATASET_TYPE == DATASET_TYPE_CHART:
        input_shape = (None, 300, 600, COLOR_CHANNELS)
        model = make_model_lstm_reversed2_chart_dense()

        '''y_train = np.zeros((x_train.shape[0], res_height, res_width, COLOR_CHANNELS), np.uint8)
        for idx_y_sample in range(len(x_train)):
            tmp = cv2.resize(x_train[idx_y_sample][0], (res_width, res_height), cv2.INTER_CUBIC)
            if DATASET_GRAY:
                tmp = np.expand_dims(tmp, -1)
            y_train[idx_y_sample] = tmp
        y_test = np.zeros((x_test.shape[0], 288, res_width, COLOR_CHANNELS), np.uint8)
        for idx_y_sample in range(len(x_test)):
            tmp = cv2.resize(x_test[idx_y_sample][0], (res_width, res_height), cv2.INTER_CUBIC)
            if DATASET_GRAY:
                tmp = np.expand_dims(tmp, -1)
            y_test[idx_y_sample] = tmp'''

        '''y_train_temp = np.zeros((x_train.shape[0], 288, 576, 3), np.uint8)
        for idx_y_sample in range(len(x_train)):
            y_train_temp[idx_y_sample] = cv2.resize(y_train[idx_y_sample], (576, 288), cv2.INTER_CUBIC)
        y_train = y_train_temp
        y_test_temp = np.zeros((x_test.shape[0], 288, 576, 3), np.uint8)
        for idx_y_sample in range(len(x_test)):
            y_test_temp[idx_y_sample] = cv2.resize(y_test[idx_y_sample], (576, 288), cv2.INTER_CUBIC)
        y_test = y_test_temp'''

        y_train = np.where(y_train == 1, 1, 0).astype(np.uint8)
        y_test = np.where(y_test == 1, 1, 0).astype(np.uint8)

# print(y_train2 - y_train)
# y_train_diff = y_train2 - y_train
# print(np.count_nonzero(y_train_diff))

model.summary()

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2)

score = model.evaluate(x_test, y_test, verbose=0)
print(score)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])

predictions = model.predict(x_test)
print("predictions shape:", predictions.shape)

if DATASET_TYPE == DATASET_TYPE_MNIST:
    cv2.imshow("test-in", (x_test[0][0]).astype(np.uint8))
    # cv2.imshow("test-in", (x_test[0]).astype(np.uint8))
    cv2.imshow("test-out", (predictions[0]*255).astype(np.uint8))

if DATASET_TYPE == DATASET_TYPE_CHART:
    # step = 1
    step = 2
    tmp1 = cv2.normalize(predictions[0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    tmp2 = cv2.normalize(predictions[1 * step], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    tmp3 = cv2.normalize(predictions[2 * step], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    tmp4 = cv2.normalize(predictions[3 * step], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if DATASET_GRAY:
        tmp1 = np.expand_dims(tmp1, -1)
        tmp2 = np.expand_dims(tmp2, -1)
        tmp3 = np.expand_dims(tmp3, -1)
        tmp4 = np.expand_dims(tmp4, -1)
    cv2.imshow("test-in0", (x_test[0][0]*255).astype(np.uint8))
    cv2.imshow("test-in1", (x_test[1 * step][0]*255).astype(np.uint8))
    cv2.imshow("test-in2", (x_test[2 * step][0]*255).astype(np.uint8))
    cv2.imshow("test-in3", (x_test[3 * step][0]*255).astype(np.uint8))
    cv2.imshow("test-out0", (tmp1*255).astype(np.uint8))
    cv2.imshow("test-out1", (tmp2*255).astype(np.uint8))
    cv2.imshow("test-out2", (tmp3*255).astype(np.uint8))
    cv2.imshow("test-out3", (tmp4*255).astype(np.uint8))

k = cv2.waitKey(0)

cv2.destroyAllWindows()