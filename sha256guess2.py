import gc
import hashlib, random, time, numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, concatenate, Flatten

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def access_bit(data, num):
    base = int(num // 8)
    shift = int(num % 8)
    return (data[base] & (1 << shift)) >> shift

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpus[0], [
#             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)

# h = hashlib.sha256()

# SAMPLES_TOTAL = 1000000
SAMPLES_TOTAL = 10000
SAMPLES_VAL_RATIO = 20
SAMPLES_TEST_RATIO = 20

X = np.zeros((SAMPLES_TOTAL, 32), np.float32)
# y = np.zeros((SAMPLES_TOTAL, 1), np.bool)
y = np.zeros((SAMPLES_TOTAL, 1), np.float32)

start = time.time()

can_add_false = False

set_size = 0
for _ in range(SAMPLES_TOTAL):
    # v = (2 ** 128 - 1)
    v = int(random.random() * (2 ** 32 - 1))
    v_bytes = v.to_bytes(4, 'little', signed=False)
    v_arr = np.frombuffer(v_bytes, dtype=np.uint8)
    v_arr_bit = np.unpackbits(v_arr, bitorder='little')
    # if v_arr in x:
    #     continue
    # h = hashlib.sha256(b"abc")
    h = hashlib.sha256(v_bytes)
    first_byte = h.digest()[0]
    if (first_byte < 127) and (not can_add_false):
        # X[set_size] = v_arr
        # X[set_size] = np.array([access_bit(v_bytes, i) for i in range(len(v_bytes)*8)])
        X[set_size] = v_arr_bit
        y[set_size] = 1
        set_size += 1
        can_add_false = True
    if (first_byte >= 127) and can_add_false:
        # X[set_size] = v_arr
        X[set_size] = v_arr_bit
        y[set_size] = 0
        set_size += 1
        can_add_false = False

end = time.time()

print(f"Runtime of dataset prep is {end - start}")

X = (X[:set_size]).astype(np.float32) / 255
y = y[:set_size]

SAMPLES_VAL = int(set_size * (SAMPLES_VAL_RATIO / 100))
SAMPLES_TEST = int(set_size * (SAMPLES_TEST_RATIO / 100))
SAMPLES_TRAIN = set_size - SAMPLES_VAL - SAMPLES_TEST

X_train = X[:SAMPLES_TRAIN]
X_val = X[SAMPLES_TRAIN:SAMPLES_TRAIN + SAMPLES_VAL]
X_test = X[-SAMPLES_TEST:]

y_train = y[:SAMPLES_TRAIN]
y_val = y[SAMPLES_TRAIN:SAMPLES_TRAIN + SAMPLES_VAL]
y_test = y[-SAMPLES_TEST:]

def makeModel2():
    model = Sequential()
    # model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))
    # model.add(LSTM(32, return_sequences=True))
    model.add(Dense(10000, activation='relu', input_shape=(4,)))
    # model.add(Dropout(0.5))
    for _ in range(30):
        model.add(Dense(10000, activation='relu'))
        # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # print(model.summary())

    # model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    model.compile(optimizer=(keras.optimizers.Adam(1e-3)), loss="binary_crossentropy", metrics=["accuracy"])

    return model

def makeModelSig():
    inputs = keras.Input(shape=(4,))

    x = inputs

    x = Dense(10000, activation='sigmoid')(x)
    # x = Dropout(0.5)(x)

    for _ in range(300):
        x = Dense(1500, activation="sigmoid")(x)
        # x = Dropout(0.5)(x)
        x = concatenate([x, inputs])
        x = Flatten()(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)

    # model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    model.compile(optimizer=(keras.optimizers.Adam(1e-3)), loss="binary_crossentropy", metrics=["accuracy"])

    return model

def makeModelBig():
    inputs = keras.Input(shape=(4,))

    x = inputs

    x = Dense(100000, activation='relu')(x)
    # x = Dropout(0.5)(x)

    for _ in range(300):
        x = Dense(1000, activation="sigmoid")(x)
        # x = Dropout(0.5)(x)
        x = concatenate([x, inputs])
        x = Flatten()(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)

    # model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    model.compile(optimizer=(keras.optimizers.Adam(1e-3)), loss="binary_crossentropy", metrics=["accuracy"])

    return model

def makeModel():
    inputs = keras.Input(shape=(32,))

    x = inputs

    x = Dense(10000, activation='sigmoid')(x)
    # x = Dropout(0.5)(x)

    for _ in range(300):
        x = Dense(1500, activation="sigmoid")(x)
        # x = Dropout(0.5)(x)
        x = concatenate([x, inputs])
        x = Flatten()(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    # model.compile(optimizer=(keras.optimizers.Adam(1e-3)), loss="binary_crossentropy", metrics=["accuracy"])

    return model

# model.fit(X, y, batch_size=1, epochs=1000)

for _ in range(10):
    model = makeModel()
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=1,
        epochs=10,
        validation_data=(X_val, y_val),
        shuffle=True
    )
    del model
    model = None
    del history
    history = None
    gc.collect()
    time.sleep(1)
