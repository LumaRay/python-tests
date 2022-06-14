import math
from datetime import datetime
import gc
import hashlib, random, time, numpy as np
import os
import multiprocessing as mp
# from multiprocessing import Process  # , shared_memory
import ctypes as c
# from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, concatenate, Flatten
from tensorflow import keras

SAVE_MODELS = True

# EPOCHS = 10
EPOCHS = 2

ROUNDS = 100

# BATCH_SIZE = 1
# BATCH_SIZE = 32
BATCH_SIZE = 64

# SAMPLES_TOTAL = 1000000
# SAMPLES_TOTAL = 10000
SAMPLES_TOTAL = 1000
# SAMPLES_TOTAL = 100
# SAMPLES_TOTAL = 10
SAMPLES_VAL_RATIO = 20
SAMPLES_TEST_RATIO = 20

'''PAYLOAD_BYTES_SIZE = 29
NONCE_BYTES_SIZE = 3
ZERO_BITS_COUNT = 15
Y_RESOLUTION_DECREASE_RATIO = 2 ** 10'''

PAYLOAD_BYTES_SIZE = 30
NONCE_BYTES_SIZE = 2
ZERO_BITS_COUNT = 11
Y_RESOLUTION_DECREASE_RATIO = 2 ** 10

'''PAYLOAD_BYTES_SIZE = 30
NONCE_BYTES_SIZE = 2
ZERO_BITS_COUNT = 8
Y_RESOLUTION_DECREASE_RATIO = 2 ** 0'''


input_size = PAYLOAD_BYTES_SIZE * 8
output_full_size = 2 ** (NONCE_BYTES_SIZE * 8)
output_size = int(output_full_size / Y_RESOLUTION_DECREASE_RATIO)

def access_bit(data, num):
    base = int(num // 8)
    shift = int(num % 8)
    return (data[base] & (1 << shift)) >> shift

def makeModel1():
    inputs = keras.Input(shape=(input_size,))

    x = inputs

    x = Dense(100, activation='sigmoid')(x)
    # x = Dropout(0.5)(x)

    for _ in range(300):
        x = Dense(150, activation="sigmoid")(x)
        # x = Dropout(0.5)(x)
        x = concatenate([x, inputs])
        x = Flatten()(x)

    outputs = Dense(output_size, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    # model.compile(optimizer=(keras.optimizers.Adam(1e-3)), loss="binary_crossentropy", metrics=["accuracy"])

    return model

def makeModel2():
    inputs = keras.Input(shape=(input_size,))

    x = inputs

    x = Dense(100, activation='relu')(x)
    # x = Dropout(0.5)(x)

    for _ in range(300):
        x = Dense(150, activation="relu")(x)
        # x = Dropout(0.5)(x)
        x = concatenate([x, inputs])
        x = Flatten()(x)

    outputs = Dense(output_size, activation='relu')(x)

    model = keras.Model(inputs, outputs)

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    # model.compile(optimizer=(keras.optimizers.Adam(1e-3)), loss="binary_crossentropy", metrics=["accuracy"])

    return model

def makeModel3():
    inputs = keras.Input(shape=(input_size,))

    x = inputs

    x = Dense(100, activation='sigmoid')(x)
    # x = Dropout(0.5)(x)

    x_prev = x

    for _ in range(300):
        x = Dense(150, activation="sigmoid")(x)
        x_dense = x
        # x = Dropout(0.5)(x)
        x = concatenate([x_dense, x_prev, inputs])
        # x = Flatten()(x)
        x_prev = x_dense

    outputs = Dense(output_size, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    # model.compile(optimizer=(keras.optimizers.Adam(1e-3)), loss="binary_crossentropy", metrics=["accuracy"])

    return model

def makeModel4():  #  va=1.0
    inputs = keras.Input(shape=(input_size,))

    x = inputs

    x = Dense(100, activation='sigmoid')(x)
    # x = Dropout(0.5)(x)

    x_prev = concatenate([x, inputs])

    for _ in range(300):
        x_dense = Dense(100, activation="sigmoid")(x)
        # x = Dropout(0.5)(x)
        x = concatenate([x_dense, x_prev])
        # x = Flatten()(x)
        x_prev = x

    outputs = Dense(output_size, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    # model.compile(optimizer=(keras.optimizers.Adam(1e-3)), loss="binary_crossentropy", metrics=["accuracy"])

    return model

def makeModel5():  #  va=1.0
    inputs = keras.Input(shape=(input_size,))

    x = inputs

    x = Dense(100, activation='sigmoid')(x)
    # x = Dropout(0.5)(x)

    x_prev = concatenate([x, inputs])

    for _ in range(100):
        x_dense = Dense(300, activation="sigmoid")(x)
        # x = Dropout(0.5)(x)
        x = concatenate([x_dense, x_prev])
        # x = Flatten()(x)
        x_prev = x

    outputs = Dense(output_size, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    # model.compile(optimizer=(keras.optimizers.Adam(1e-3)), loss="binary_crossentropy", metrics=["accuracy"])

    return model

def makeModel():
    inputs = keras.Input(shape=(input_size,))

    x = inputs
    x_prev = x

    for _ in range(300):
        x_dense = Dense(100, activation="sigmoid")(x)
        # x = Dropout(0.5)(x)
        x = concatenate([x_dense, x_prev])
        # x = Flatten()(x)
        x_prev = x

    outputs = Dense(output_size, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    # model.compile(optimizer=(keras.optimizers.Adam(1e-3)), loss="binary_crossentropy", metrics=["accuracy"])

    return model

# model.fit(X, y, batch_size=1, epochs=1000)

def generateTask(mp_X_arr, mp_y_arr, cpu_idx, cpu_count):

    X = np.frombuffer(mp_X_arr.get_obj())
    X = X.reshape((SAMPLES_TOTAL, input_size))
    y = np.frombuffer(mp_y_arr.get_obj())
    y = y.reshape((SAMPLES_TOTAL, output_size))

    for sample_idx in range(math.ceil(SAMPLES_TOTAL / cpu_count)):
        sample_idx = sample_idx * cpu_count + cpu_idx
        if sample_idx >= SAMPLES_TOTAL:
            break
        # v = (2 ** 128 - 1)
        # v = int(random.random() * (2 ** 32 - 1))
        # v_bytes = v.to_bytes(4, 'little', signed=False)
        payload1_bytes = os.urandom(PAYLOAD_BYTES_SIZE)
        payload1_arr = np.frombuffer(payload1_bytes, dtype=np.uint8)
        payload1_arr_bits = np.unpackbits(payload1_arr, bitorder='little')
        X_sample = payload1_arr_bits.astype(np.float32)
        y_sample = np.zeros(output_size, np.float32)
        payload2_idx = 0
        for _ in range(output_full_size):
            # payload2_bytes = os.urandom(4)
            payload2_bytes = payload2_idx.to_bytes(4, 'little', signed=False)
            payload2_arr = np.frombuffer(payload2_bytes, dtype=np.uint8)
            # payload2_arr_bits = np.unpackbits(payload2_arr, bitorder='little')
            payload_bytes = payload1_bytes + payload2_bytes
            h = hashlib.sha256(payload_bytes)
            h_digest_bytes = h.digest()
            h_digest_arr = np.frombuffer(h_digest_bytes, dtype=np.uint8)
            digest_arr_bits = np.unpackbits(h_digest_arr, bitorder='little')
            if not np.any(digest_arr_bits[:ZERO_BITS_COUNT]):
                y_sample_idx = int(payload2_idx / Y_RESOLUTION_DECREASE_RATIO)
                y_sample[y_sample_idx] = 1
                payload2_idx = (y_sample_idx + 1) * Y_RESOLUTION_DECREASE_RATIO
            else:
                payload2_idx += 1
            if payload2_idx >= output_full_size:
                break
        X[sample_idx] = X_sample
        y[sample_idx] = y_sample

def testTask(mp_X_arr, mp_y_arr, idx_round, output_test_path):
    SAMPLES_VAL = int(SAMPLES_TOTAL * (SAMPLES_VAL_RATIO / 100))
    SAMPLES_TEST = int(SAMPLES_TOTAL * (SAMPLES_TEST_RATIO / 100))
    SAMPLES_TRAIN = SAMPLES_TOTAL - SAMPLES_VAL - SAMPLES_TEST

    import tensorflow as tf
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    X = np.frombuffer(mp_X_arr.get_obj())
    X = X.reshape((SAMPLES_TOTAL, input_size))
    y = np.frombuffer(mp_y_arr.get_obj())
    y = y.reshape((SAMPLES_TOTAL, output_size))

    X_train = X[:SAMPLES_TRAIN]
    X_val = X[SAMPLES_TRAIN:SAMPLES_TRAIN + SAMPLES_VAL]
    X_test = X[-SAMPLES_TEST:]

    y_train = y[:SAMPLES_TRAIN]
    y_val = y[SAMPLES_TRAIN:SAMPLES_TRAIN + SAMPLES_VAL]
    y_test = y[-SAMPLES_TEST:]

    model = makeModel4()
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        shuffle=True
    )
    train_loss, val_loss, train_accuracy, val_accuracy = history.history['loss'], history.history['val_loss'], history.history['accuracy'], history.history['val_accuracy']
    max_val_accuracy = max(val_accuracy)
    min_val_loss = min(val_loss)
    results_test = model.evaluate(X_test, y_test, batch_size=1)
    print("results after fit test loss, test acc:", results_test)
    if SAVE_MODELS:
        try:
            model.save(output_test_path + "/sha256_ta{0:03d}".format(
                int(round(results_test[1] * 1000, 0))) + "tl{0:03d}".format(
                int(round(results_test[0] * 1000, 0))) + "_i_va{:.5f}".format(
                round(max_val_accuracy, 3)) + "vl{:.5f}".format(round(min_val_loss, 3)) + "_r{0:02d}".format(
                idx_round + 1) + "e{0:04d}".format(EPOCHS) + '.h5')
        except:
            pass
    del model
    model = None
    del history
    history = None
    gc.collect()
    time.sleep(1)


# multiprocessing.cpu_count()

if __name__ == '__main__':
    OUTPUT_TEST_PATH = "G:/output/sha256-tests/" + datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    if not os.path.exists(OUTPUT_TEST_PATH):
        os.makedirs(OUTPUT_TEST_PATH)

    # sha Input vector = 32 bytes: 28 bytes payload + 4 bytes nonce
    # sha Output has length of zeros of 8 bit size (8 float lanes) or 1 float size (1 float lane) or none if zero bits count is static

    mp_X_arr = mp.Array(c.c_double, SAMPLES_TOTAL * input_size)
    mp_y_arr = mp.Array(c.c_double, SAMPLES_TOTAL * output_size)
    X = np.frombuffer(mp_X_arr.get_obj())
    X = X.reshape((SAMPLES_TOTAL, input_size))
    y = np.frombuffer(mp_y_arr.get_obj())
    y = y.reshape((SAMPLES_TOTAL, output_size))

    # X = np.zeros((SAMPLES_TOTAL, input_size), np.float32)
    # y = np.zeros((SAMPLES_TOTAL, output_size), np.float32)

    cpu_count = mp.cpu_count()

    print(f"Found {cpu_count} CPUs")

    start = time.time()

    '''p = mp.Pool(processes=cpu_count)
    p.starmap(generateTask, [(mp_X_arr, mp_y_arr, cpu_idx, cpu_count,) for cpu_idx in range(cpu_count)])
    p.close()
    p.join()'''

    gen_processes = []
    for cpu_idx in range(cpu_count):
        p = mp.Process(target=generateTask, args=(mp_X_arr, mp_y_arr, cpu_idx, cpu_count,))
        p.start()
        gen_processes += [p]

    for p in gen_processes:
        p.join()

    for p in gen_processes:
        p.close()

    # can_add_false = False

    '''for sample_idx in range(SAMPLES_TOTAL):
        # v = (2 ** 128 - 1)
        # v = int(random.random() * (2 ** 32 - 1))
        # v_bytes = v.to_bytes(4, 'little', signed=False)
        payload1_bytes = os.urandom(PAYLOAD_BYTES_SIZE)
        payload1_arr = np.frombuffer(payload1_bytes, dtype=np.uint8)
        payload1_arr_bits = np.unpackbits(payload1_arr, bitorder='little')
        X_sample = payload1_arr_bits.astype(np.float32)
        y_sample = np.zeros(output_size, np.float32)
        payload2_idx = 0
        for _ in range(output_full_size):
            # payload2_bytes = os.urandom(4)
            payload2_bytes = payload2_idx.to_bytes(4, 'little', signed=False)
            payload2_arr = np.frombuffer(payload2_bytes, dtype=np.uint8)
            # payload2_arr_bits = np.unpackbits(payload2_arr, bitorder='little')
            payload_bytes = payload1_bytes + payload2_bytes
            h = hashlib.sha256(payload_bytes)
            h_digest_bytes = h.digest()
            h_digest_arr = np.frombuffer(h_digest_bytes, dtype=np.uint8)
            digest_arr_bits = np.unpackbits(h_digest_arr, bitorder='little')
            if not np.any(digest_arr_bits[:ZERO_BITS_COUNT]):
                y_sample_idx = int(payload2_idx / Y_RESOLUTION_DECREASE_RATIO)
                y_sample[y_sample_idx] = 1
                payload2_idx = (y_sample_idx + 1) * Y_RESOLUTION_DECREASE_RATIO
            else:
                payload2_idx += 1
            if payload2_idx >= output_full_size:
                break
        X[sample_idx] = X_sample
        y[sample_idx] = y_sample'''

    end = time.time()

    print(f"Runtime of dataset prep is {end - start}")  #  Runtime of dataset prep is 7736.425572872162 for 10000 no paralleling   1776.835440158844 for 10000 paralleling   x4.35 faster

    for idx_round in range(ROUNDS):
        # testTask(mp_X_arr, mp_y_arr, idx_round, OUTPUT_TEST_PATH)
        p = mp.Process(target=testTask, args=(mp_X_arr, mp_y_arr, idx_round, OUTPUT_TEST_PATH,))
        p.start()
        p.join()
        p.close()
        del p
        gc.collect()


'''import ctypes as c
import numpy as np
import multiprocessing as mp

n, m = 2, 3
mp_arr = mp.Array(c.c_double, n*m) # shared, can be used from multiple processes
# then in each new process create a new numpy array using:
arr = np.frombuffer(mp_arr.get_obj()) # mp_arr and arr share the same memory
# make it two-dimensional
b = arr.reshape((n,m)) # b and arr share the same memory'''
