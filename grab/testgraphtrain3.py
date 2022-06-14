import time
from mss import mss
# from PIL import Image
import numpy as np
import cv2
from datetime import datetime
import os
import pathlib
pathToScriptFolder = str(pathlib.Path().absolute())

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ConvLSTM2D, Activation, MaxPooling3D, AveragePooling3D, Dense, Dropout, Reshape, LSTM, Flatten, BatchNormalization, Input, concatenate, UpSampling2D, Conv2D, GlobalAveragePooling2D
import tensorflow as tf

from matplotlib import pyplot as plt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

IMPORT_FOLDER = pathToScriptFolder + f"/frames_src/simple-line-10s-span10min-es2/2021-05-11-12-27-38-361400/"
MODEL_FOLDER = pathToScriptFolder + "/simple-line-10s-span10min-es2/" + train_timestamp + "/"
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

buy_width, buy_height = 170, 310
sell_width, sell_height = 170, 320
hist_width, hist_height = 170, 710
chart_width, chart_height = 600, 300
# res_height, res_width = 248, 136
# res_width, res_height = 480, 240
res_width, res_height = 576, 288

COLOR_CHANNELS = 3

MAX_SAMPLES = 5000


HISTORICAL_FRAMES = 1

NUM_EPOCHS = 10
BATCH_SIZE = 1
VALID_RATIO = 25
TEST_RATIO = 5
lr = 0.0008

def createImageInputModel(input_shape):
    # from keras import backend as K
    # K.set_image_data_format('channels_first')
    model_input = Input(shape=input_shape)
    # x = ConvLSTM2D(32, kernel_size=(7, 7), padding='valid', return_sequences=True)(model_input)
    x = ConvLSTM2D(48, kernel_size=(3, 3), padding='same', return_sequences=True, data_format='channels_first')(model_input)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), data_format='channels_first')(x)

    # nfeatures = 32
    # nconvs = 0
    while (x.shape[-1] > 10 or x.shape[-2] > 10):
        x = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True, data_format='channels_first')(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), data_format='channels_first')(x)
        # nfeatures += 32
        # nconvs += 1

    # x = GlobalAveragePooling2D()(x)
    # x = Reshape((x.get_shape()[0], x.get_shape()[1], x.get_shape()[2] * x.get_shape()[3]))(x)
    x = Flatten()(x)
    '''x_shape1 = x.shape[1]
    # x = Dense(320)(x)
    x = Dense(x_shape1)(x)
    # x = Flatten()(x)
    # x = Dense(1)(x)
    # x = Dense(32)(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.5)(x)'''
    # x = Flatten()(x)
    model = Model(model_input, x)
    return model_input, model

def createMergedInputModel():

    input_buy, model_buy = createImageInputModel((HISTORICAL_FRAMES, COLOR_CHANNELS, buy_height, buy_width))
    input_sell, model_sell = createImageInputModel((HISTORICAL_FRAMES, COLOR_CHANNELS, sell_height, sell_width))
    input_hist, model_hist = createImageInputModel((HISTORICAL_FRAMES, COLOR_CHANNELS, hist_height, hist_width))
    input_chart, model_chart = createImageInputModel((HISTORICAL_FRAMES, COLOR_CHANNELS, chart_height, chart_width))

    '''out_shape = model.output_shape
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

    combined = concatenate([model_buy.output, model_sell.output, model_hist.output, model_chart.output])
    # combined = concatenate([model_buy.output, model_sell.output])  #  , model_3.output, model_4.output, model_5.output, model_6.output])
    # combined = model_chart.output
    x = combined
    # x = Flatten()(x)
    # x = Dense(4*64 * 64, activation="sigmoid")(x)
    #x = Dropout(0.1)(x)
    # x = Dense(64 * 64, activation="sigmoid")(x)
    #x = Dropout(0.1)(x)
    # x = Dense(64 * 64, activation="sigmoid")(x)
    #x = Dropout(0.1)(x)
    # x = Dense(64 * 64, activation="sigmoid")(x)
    #x = Dropout(0.1)(x)
    # x = Dense(64*64, activation="sigmoid")(x)
    #x = Dropout(0.1)(x)
    # x = Dense(64*32, activation="sigmoid")(x)
    # x = Dense(int(buy_height/10)*int(buy_width/10)*3, activation="sigmoid")(x)
    # x = Dense(int(chart_height/10)*int(chart_width/10)*3, activation="sigmoid")(x)
    x = Dense(int(chart_height/32)*int(chart_width/32)*64, activation="sigmoid")(x)
    x = Activation("sigmoid")(x)
    x = Dropout(0.5)(x)
    #x = Dropout(0.1)(x)
    # x = Reshape((int(buy_height/10), int(buy_width/10), 3))(x)
    # x = Reshape((int(chart_height/10), int(chart_width/10), 3))(x)
    x = Reshape((int(chart_height/32), int(chart_width/32), 64))(x)
    #x = Dense(1, activation="linear")(x)
    # x = Dense(1, activation="sigmoid")(x)
# , data_format='channels_first'
    # https://github.com/HackerPoet/DeepDoodle/blob/master/train.py
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    # x = Conv2D(384, (3, 3), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    # x = Conv2D(192, (3, 3), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    # x = Conv2D(96, (3, 3), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(48, (3, 3), padding='same')(x)
    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(24, (3, 3), padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(3, (1, 1), padding='same')(x)
    x = Activation("sigmoid")(x)

    # model.compile(optimizer=Adam(lr=lr), loss='mse')
    # plot_model(model, to_file='model.png', show_shapes=True)


    # model = Model(inputs=[model_buy.input], outputs=x)  # , model_2.input, model_3.input, model_4.input, model_5.input, model_6.input], outputs=x)
    # model = Model(inputs=[model_buy.input, model_sell.input], outputs=x)  # , model_3.input, model_4.input, model_5.input, model_6.input], outputs=x)
    model = Model(inputs=[model_buy.input, model_sell.input, model_hist.input, model_chart.input], outputs=x)

    return model

model = createMergedInputModel()
#opt = keras.optimizers.Adam(lr=1e-3, decay=1e-3 / 200)
#model.compile(loss=keras.losses.mean_absolute_percentage_error, optimizer=opt)
# model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.SGD(), metrics=[keras.metrics.mae])
#model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam(), metrics=[keras.metrics.mae])
# model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam(lr=lr), metrics=[keras.metrics.mae])
model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')
print(model.summary())



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
        # frame_color = cv2.resize(frame_color, (face_width, face_height))
        frame_color = cv2.normalize(frame_color, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # frame_color = (frame_color / 256).astype(np.float32)
        if type == "buy":
            lst_buy.append(frame_color)
        if type == "sell":
            lst_sell.append(frame_color)
        if type == "hist":
            lst_hist.append(frame_color)
        if type == "chart":
            lst_chart.append(frame_color)

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
    y_samples[idx_start] = cv2.resize(lst_chart[idx_start + HISTORICAL_FRAMES], (res_width, res_height), cv2.INTER_CUBIC)

valid_samples_count = int(len(x_samples_buy)*VALID_RATIO/100)
test_samples_count = int(len(x_samples_buy)*TEST_RATIO/100)
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

y_train = np.where(y_train == 1, 1, 0)
y_test = np.where(y_test == 1, 1, 0)

'''for idx_y_sample in range(y_train.shape[0]):
    y_train[idx_y_sample] = np.where(y_train[idx_y_sample] == 1, 1, 0)
    # y_train[idx_y_sample] = cv2.resize(y_train[idx_y_sample], (y_train.shape[2], y_train.shape[1]), cv2.INTER_CUBIC)
    # y_train[idx_y_sample] = cv2.adaptiveThreshold(y_train[idx_y_sample], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(y_train[idx_y_sample], (5, 5), 0)
    # ret3, y_train[idx_y_sample] = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
for idx_y_sample in range(y_test.shape[0]):
    y_test[idx_y_sample] = np.where(y_test[idx_y_sample] == 1, 1, 0)
    # y_test[idx_y_sample] = cv2.resize(y_test[idx_y_sample], (y_test.shape[2], y_test.shape[1]), cv2.INTER_CUBIC)
    # y_test[idx_y_sample] = cv2.adaptiveThreshold(y_test[idx_y_sample], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(y_test[idx_y_sample], (5, 5), 0)
    # ret3, y_test[idx_y_sample] = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)'''

path_checkpoint = MODEL_FOLDER + "model_checkpoint.h5"
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
    x=[np.moveaxis(x_train_buy, -1, 2), np.moveaxis(x_train_sell, -1, 2), np.moveaxis(x_train_hist, -1, 2), np.moveaxis(x_train_chart, -1, 2)],
    y=y_train,
    batch_size=BATCH_SIZE,#5,#10,#64,
    epochs=NUM_EPOCHS,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=([np.moveaxis(x_valid_buy, -1, 2), np.moveaxis(x_valid_sell, -1, 2), np.moveaxis(x_valid_hist, -1, 2), np.moveaxis(x_valid_chart, -1, 2)], y_valid),
    verbose=1,
    shuffle=True,
    callbacks=[es_callback],#, modelckpt_callback],
    #validation_split=0.2,
)

'''try:
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
    pass'''


results = model.evaluate([np.moveaxis(x_test_buy, -1, 2), np.moveaxis(x_test_sell, -1, 2), np.moveaxis(x_test_hist, -1, 2), np.moveaxis(x_test_chart, -1, 2)], y_test, batch_size=1)#128)
print("test loss, test acc:", results)

model_path = MODEL_FOLDER + "gpaph_" + train_timestamp + "{:.3f}".format(round(results, 3)) + ".h5"
model.save(model_path)

predictions = model.predict([np.moveaxis(x_test_buy, -1, 2), np.moveaxis(x_test_sell, -1, 2), np.moveaxis(x_test_hist, -1, 2), np.moveaxis(x_test_chart, -1, 2)])
print("predictions shape:", predictions.shape)
# print("prediction [0, 0]:", predictions[0, 0] * MAX_TEMP + MIN_TEMP, " should be:", y_test[0, 0, 0] * MAX_TEMP + MIN_TEMP)

predictions[0] = cv2.normalize(predictions[0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
cv2.imshow("test-in", (x_test_chart[0][0] * 255).astype(np.uint8))
cv2.imshow("test-out", (predictions[0] * 255).astype(np.uint8))
# cv2.imshow("test", (predictions[0]*255).astype(np.uint8))

k = cv2.waitKey(0)

# input("Press Enter to continue...")

cv2.destroyAllWindows()

