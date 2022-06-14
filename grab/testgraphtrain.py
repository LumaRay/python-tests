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
from tensorflow.keras.layers import ConvLSTM2D, Activation, MaxPooling3D, Dense, Dropout, Reshape, LSTM, Flatten, BatchNormalization, Input, concatenate, UpSampling2D, Conv2D
import tensorflow as tf

from matplotlib import pyplot as plt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

IMPORT_FOLDER = pathToScriptFolder + f"/../frames_src/simple-line-10s-span10min-es2/2021-05-11-12-27-38-361400/"
MODEL_FOLDER = pathToScriptFolder + "/../simple-line-10s-span10min-es2/" + train_timestamp + "/"

buy_width, buy_height = 170, 310
sell_width, sell_height = 170, 320
hist_width, hist_height = 170, 710
chart_width, chart_height = 600, 300
res_height, res_width = 248, 136
COLOR_CHANNELS = 3

MAX_SAMPLES = 5000


HISTORICAL_FRAMES = 5

NUM_EPOCHS = 2
BATCH_SIZE = 1
VALID_RATIO = 25
TEST_RATIO = 5
lr = 0.0008

lst_buy = []
lst_sell = []
lst_hist = []
lst_chart = []

files = os.listdir(IMPORT_FOLDER)
for fidx, file in enumerate(files):
    if len(lst_buy) >= MAX_SAMPLES:
        break
    if file.endswith(".jpg"):
        file_name, file_ext = file.split('.')
        # print("Parsing file " + str(fidx) + " of " + str(len(files)))
        frame_timestamp, type = file_name.split('_')
        frame_color = cv2.imread(IMPORT_FOLDER + file)
        # frame_color = cv2.resize(frame_color, (face_width, face_height))
        # frame_color = cv2.normalize(frame_color, 0, 255, cv2.NORM_MINMAX)
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

x_samples = np.zeros((total_samples, HISTORICAL_FRAMES, buy_height, buy_width, COLOR_CHANNELS), dtype=np.float32)
y_samples = np.zeros((total_samples, res_height, res_width, COLOR_CHANNELS), dtype=np.float32)

for idx_start in range(total_samples):
    x_samples[idx_start] = np.asarray([img_buy for img_buy in lst_buy[idx_start:idx_start + HISTORICAL_FRAMES]])
    y_samples[idx_start] = cv2.resize(lst_buy[idx_start + HISTORICAL_FRAMES + 1], (res_width, res_height), cv2.INTER_CUBIC)

def createInputModel(input_shape):
    model_input = Input(shape=input_shape)
    # x = ConvLSTM2D(32, kernel_size=(7, 7), padding='valid', return_sequences=True)(model_input)
    x = ConvLSTM2D(32, kernel_size=(7, 7), padding='valid', return_sequences=True, data_format='channels_first')(model_input)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = ConvLSTM2D(64, kernel_size=(5, 5), padding='valid', return_sequences=True)(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True)(x)
    x = Activation('relu')(x)
    # x = ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True)(x)
    # x = Activation('relu')(x)
    x = ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True)(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = Dense(320)(x)
    # x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    model = Model(model_input, x)
    return model_input, model

input_buy, model_buy = createInputModel((HISTORICAL_FRAMES, COLOR_CHANNELS, buy_height, buy_width))
# input_sell, model_sell = createInputModel((HISTORICAL_FRAMES, COLOR_CHANNELS, sell_height, sell_width))
# input_hist, model_hist = createInputModel((HISTORICAL_FRAMES, COLOR_CHANNELS, hist_height, hist_width))

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

# combined = concatenate([model_buy.output, model_sell.output])  #  , model_3.output, model_4.output, model_5.output, model_6.output])
combined = model_buy.output
x = combined
x = Flatten()(x)
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
x = Dense(int(buy_height/10)*int(buy_width/10)*3, activation="sigmoid")(x)
#x = Dropout(0.1)(x)
x = Reshape((int(buy_height/10), int(buy_width/10), 3))(x)
#x = Dense(1, activation="linear")(x)
# x = Dense(1, activation="sigmoid")(x)

# https://github.com/HackerPoet/DeepDoodle/blob/master/train.py
x = UpSampling2D(size=(2, 2))(x)
x = Conv2D(192, (5, 5), padding='same')(x)
x = Activation("relu")(x)
x = UpSampling2D(size=(2, 2))(x)
x = Conv2D(96, (5, 5), padding='same')(x)
x = Activation("relu")(x)
x = UpSampling2D(size=(2, 2))(x)
x = Conv2D(48, (5, 5), padding='same')(x)
x = Activation("relu")(x)
x = Conv2D(3, (1, 1), padding='same')(x)
x = Activation("sigmoid")(x)

# model.compile(optimizer=Adam(lr=lr), loss='mse')
# plot_model(model, to_file='model.png', show_shapes=True)


model = Model(inputs=[model_buy.input], outputs=x)  # , model_2.input, model_3.input, model_4.input, model_5.input, model_6.input], outputs=x)
# model = Model(inputs=[model_buy.input, model_sell.input], outputs=x)  # , model_3.input, model_4.input, model_5.input, model_6.input], outputs=x)
#opt = keras.optimizers.Adam(lr=1e-3, decay=1e-3 / 200)
#model.compile(loss=keras.losses.mean_absolute_percentage_error, optimizer=opt)
# model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.SGD(), metrics=[keras.metrics.mae])
#model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam(), metrics=[keras.metrics.mae])
model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam(lr=lr), metrics=[keras.metrics.mae])

print(model.summary())

valid_samples_count = int(len(x_samples)*VALID_RATIO/100)
test_samples_count = int(len(x_samples)*TEST_RATIO/100)
train_samples_count = len(x_samples) - valid_samples_count - test_samples_count

x_train = x_samples[:train_samples_count]
y_train = y_samples[:train_samples_count]
x_valid = x_samples[train_samples_count:-test_samples_count]
y_valid = y_samples[train_samples_count:-test_samples_count]
x_test = x_samples[-test_samples_count:]
y_test = y_samples[-test_samples_count:]

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
    x=np.moveaxis(x_train, -1, 2),
    y=y_train,
    batch_size=BATCH_SIZE,#5,#10,#64,
    epochs=NUM_EPOCHS,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(np.moveaxis(x_valid, -1, 2), y_valid),
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


results = model.evaluate(np.moveaxis(x_test, -1, 2), y_test, batch_size=1)#128)
print("test loss, test acc:", results)

predictions = model.predict(np.moveaxis(x_test, -1, 2))
print("predictions shape:", predictions.shape)
# print("prediction [0, 0]:", predictions[0, 0] * MAX_TEMP + MIN_TEMP, " should be:", y_test[0, 0, 0] * MAX_TEMP + MIN_TEMP)

cv2.imshow("test", (predictions[0]*255).astype(np.uint8))

k = cv2.waitKey(0)

# input("Press Enter to continue...")

cv2.destroyAllWindows()

