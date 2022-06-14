import os
import cv2
import json
import random
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, Activation, MaxPooling3D, Dense, Dropout, Reshape, LSTM, Flatten, BatchNormalization, Input, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# https://graphviz.org/download/
# os.environ['PATH'] = os.environ['PATH']+';'+r"F:\Work\InfraredCamera\graphviz-2.44.1-win32\Graphviz\bin"

pathToScriptFolder = str(pathlib.Path().absolute().as_posix())
dataset_sources_path = pathToScriptFolder + '/../dataset_sources'

#DATASET_NAME = 'maskfacesnewwork12falsefaceswork1nb'
#DATASET_PARSED_FOLDER = 'parsed_data_work12_false_faces_work1'

#DATASET_NAME = 'maskfacesnewwork0312added1nb'
#DATASET_PARSED_FOLDER = 'parsed_data_work0312'

#DATASET_NAME = 'maskfacesnewwork12falsefaceswork1added1nb'
#DATASET_PARSED_FOLDER = 'parsed_data_work12_false_faces_work1_added1'

#DATASET_NAME = 'maskfacesnewwork0312added1ffw1a1nb'
#DATASET_PARSED_FOLDER = 'parsed_data_work0312_false_faces_work1_added1'
#CLASS_NAMES = ['mask', 'maskchin', 'masknone', 'masknose']

#DATASET_NAME = 'maskfacesnewwork0312added1ffw1a1'
#DATASET_PARSED_FOLDER = 'parsed_data_work0312_false_faces_work1_added1'
#CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']

#DATASET_NAME = 'maskfacesnewwork0312added1ffw1a1vk1exp1'
#DATASET_PARSED_FOLDER = 'parsed_data_work0312_false_faces_work1_added1_vk_sources1exp1'

DATASET_NAMES = [
    'maskfacesnew',
    'maskfacesnew3',
    'maskfacesnewwork1',
    'maskfacesnewwork2',
    'maskfacesnewwork_toadd1',
    'false_faces_work1',
    'source_2020_11_25_1000ms',
    'source_2020_11_25_1000ms5max',
    'source_2020_11_25_200ms',
    'source_2020_11_25_3000ms',
    'source_2020_11_25_3000ms5max',
    'source_2020_11_25_300ms',
    'source_2020_11_25_noms',
    'vk_sources1',
]

CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']

MODEL_NAME = 'maskfacesnewwork0312added1ffw1a1vk1exp1'

classes_source_path_list = [dataset_sources_path + '/' + dsn for dsn in DATASET_NAMES]

#parsed_data_path = pathToScriptFolder + '/../parsed_data/' + DATASET_PARSED_FOLDER + '/'
#objects_source_path = parsed_data_path + 'objects'
#images_source_path = parsed_data_path + 'frames'
parsed_data_path = pathToScriptFolder + '/../parsed_data'
parsed_data_last_path_list = []

#FACE_WIDTH = 64
#FACE_HEIGHT = 64
FACE_WIDTH = 128
FACE_HEIGHT = 128

FACE_CHANNELS = 3

#PORTION_VAL = 30
PORTION_VAL = 20
PORTION_TEST = 1

#EPOCHS = 50
EPOCHS = 1000

MAX_CLASS_IMAGES = 999999
#MAX_CLASS_IMAGES = 100

num_classes = len(CLASS_NAMES)

in_shape = (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS)

'''TRAIN_FACES_COUNT = 500
VALID_FACES_COUNT = 60
TEST_FACES_COUNT = 11

x_train = np.zeros((TRAIN_FACES_COUNT, FACE_HEIGHT, FACE_WIDTH, FACE_CHANNELS), dtype=np.float64)
y_train = np.zeros((TRAIN_FACES_COUNT, NUM_CLASSES), dtype=np.float64)

x_val = np.zeros((VALID_FACES_COUNT, FACE_HEIGHT, FACE_WIDTH, FACE_CHANNELS), dtype=np.float64)
y_val = np.zeros((VALID_FACES_COUNT, NUM_CLASSES), dtype=np.float64)

x_test = np.zeros((TEST_FACES_COUNT, FACE_HEIGHT, FACE_WIDTH, FACE_CHANNELS), dtype=np.float64)
y_test = np.zeros((TEST_FACES_COUNT, NUM_CLASSES), dtype=np.float64)'''

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#classes_source_path = dataset_sources_path + '/' + DATASET_NAME

#dataset_outputs_path = pathToScriptFolder + '/../dataset_outputs'
#train_output_path = dataset_outputs_path + '/' + DATASET_NAME + '/train/images'
#valid_output_path = dataset_outputs_path + '/' + DATASET_NAME + '/valid/images'

#temp_path = pathToScriptFolder + '/../temp'
#if not os.path.exists(temp_path):
#    os.makedirs(temp_path)

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

def findParsedFile(file_name):
    global parsed_data_path, parsed_data_last_path_list
    for parsed_data_last_path in parsed_data_last_path_list:
        check_path = os.path.join(parsed_data_last_path, file_name)
        if os.path.isfile(check_path):
            return check_path
    for root, subdirs, files in os.walk(parsed_data_path):
        if file_name in files:
            parsed_data_last_path_list.insert(0, root)
            return os.path.join(root, file_name)
    return None

def loadClassesTodatasets(classes_source_path_list):
    #class_pathes = [f.path for f in os.scandir(classes_source_path) if f.is_dir()]

    #dataset = np.zeros((len(class_pathes), FACE_HEIGHT, FACE_WIDTH, FACE_CHANNELS), dtype=np.float64)

    lst_x_dataset = []
    lst_y_dataset = []
    '''lst_y_names = []

    for class_path in class_pathes:
        class_name = os.path.basename(os.path.normpath(class_path))
        try:
            if lst_y_names.index(class_name) == -1:
                lst_y_names.append(class_name)
        except:
            lst_y_names.append(class_name)'''

    #for class_path in class_pathes:
    for class_idx, class_name in enumerate(CLASS_NAMES):
        for classes_source_path in classes_source_path_list:
            #class_name = os.path.basename(os.path.normpath(class_path))
            print("Parsing class " + class_name)
            #class_idx = CLASS_NAMES.index(class_name)
            class_path = classes_source_path + '/' + class_name
            files_list = os.listdir(class_path)
            for fidx, file in enumerate(files_list):
                if fidx > MAX_CLASS_IMAGES:
                    break
                if file.endswith(".jpg"):
                    file_name, file_ext = file.split('.')
                    print("Parsing class " + class_name + " with file " + str(fidx) + " of " + str(len(files_list)))
                    #frame_timestamp, score, timestamp = file_name.split('_')
                    object_path = findParsedFile(file_name + '.txt')
                    #with open(objects_source_path + '/' + file_name + '.txt') as json_file:
                    with open(object_path) as json_file:
                        entry = json.load(json_file)
                        frame_ref = entry['frame_ref']
                        #frame_width = entry['frame_width']
                        #frame_height = entry['frame_height']
                        #contour_area = entry['contour_area']
                        bbox = entry['bbox']
                        bbox_x = bbox['x']
                        bbox_y = bbox['y']
                        bbox_w = bbox['w']
                        bbox_h = bbox['h']
                        contour = entry['contour']
                        np_contour = np.zeros((len(contour), 2), dtype=np.int32)
                        '''for i, contour_point in enumerate(contour):
                            x = contour_point['x']
                            y = contour_point['y']
                            np_contour[i] = [x, y]
                        img = cv2.imread(images_source_path + '/' + frame_ref + ".jpg")
                        mask = np.zeros(img.shape, np.uint8)
                        mask.fill(127)
                        cv2.drawContours(mask, [np_contour], -1, (255), 1)
                        res = cv2.bitwise_and(img, img, mask=mask)
                        res_roi = res[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]'''
                        for i, contour_point in enumerate(contour):
                            x = contour_point['x'] - bbox_x
                            y = contour_point['y'] - bbox_y
                            np_contour[i] = [x, y]
                        #frame_path = images_source_path + '/' + frame_ref + ".jpg"
                        frame_path = findParsedFile(frame_ref + ".jpg")
                        try:
                            img = cv2.imread(frame_path)
                            img_roi = img[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
                            mask_roi = np.zeros(img_roi.shape[:2], np.uint8)
                            cv2.drawContours(mask_roi, [np_contour], -1, (255), thickness=-1)
                            res_roi = cv2.bitwise_and(img_roi, img_roi, mask=mask_roi)
                            mask_roi -= 255
                            gray_roi = np.zeros(img_roi.shape, np.uint8)
                            gray_roi.fill(127)
                            gray_roi_reversed = cv2.bitwise_and(gray_roi, gray_roi, mask=mask_roi)
                            res_roi = cv2.addWeighted(res_roi, 1, gray_roi_reversed, 1, 0)
                            res_roi = cv2.resize(res_roi, (FACE_WIDTH, FACE_HEIGHT), cv2.INTER_CUBIC)
                            #cv2.imshow("123", res_roi)
                            #cv2.imwrite(temp_path + '/' + file_name + '_face_roi.jpg', res_roi)
                            #res_roi = res_roi.reshape((1, 3, ) + res_roi.shape[:2])  # this is a Numpy array with shape (1, 3, 150, 150)
                            #res_roi = res_roi.reshape((3,) + res_roi.shape[:2])  # this is a Numpy array with shape (3, 150, 150)
                            #res_roi = res_roi.astype(np.float) # / 255
                            res_roi = res_roi.copy()
                            lst_x_dataset.append(res_roi)
                            #lst_res_roi = datagen.flow(res_roi, batch_size=1)  # , save_to_dir='preview', save_prefix='cat', save_format='jpeg')
                            #lst_x_dataset.append(lst_res_roi)
                            y_set = np.zeros(num_classes)
                            y_set[class_idx] = 1
                            lst_y_dataset.append(y_set)
                        except:
                            print("Failed to process " + frame_path + " !!!")

    #x_train = x_train.map(lambda x, y: (data_augmentation(x, training=True), y))

    tmp = list(zip(lst_x_dataset, lst_y_dataset))

    random.shuffle(tmp)

    lst_x_dataset, lst_y_dataset = zip(*tmp)

    return (np.array(lst_x_dataset), np.array(lst_y_dataset))

#x_out, y_out = loadClassesTodatasets(classes_source_path)
x_out, y_out = loadClassesTodatasets(classes_source_path_list)

count_train = int(len(x_out) * (100 - PORTION_VAL - PORTION_TEST) / 100)
count_val = int(len(x_out) * PORTION_VAL / 100)
count_test = int(len(x_out) * PORTION_TEST / 100)

x_train = x_out[:count_train]
y_train = y_out[:count_train]
x_val = x_out[count_train:count_train+count_val]
y_val = y_out[count_train:count_train+count_val]
x_test = x_out[-count_test:]
y_test = y_out[-count_test:]

#x_val, y_val = loadClassesTodatasets(valid_output_path)

#contours = [np.array([[1, 1], [10, 50], [50, 50]], dtype=np.int32), np.array([[99, 99], [99, 60], [60, 99]], dtype=np.int32)]

#drawing = np.zeros([100, 100], np.uint8)
#for cnt in contours:
#    cv2.drawContours(drawing, [cnt], 0, (255, 255, 255), 2)

#cv2.imshow('output', drawing)
#cv2.waitKey(0)


'''def experiment1():
    def createModel1():
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=(FACE_CHANNELS, 150, 150)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model

    model1 = createModel1()

    batch_size = 16

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    model1.fit_generator(
            train_generator,
            steps_per_epoch=2000 // batch_size,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800 // batch_size)
    model1.save_weights('first_try.h5')

    model1.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    model1.fit(train_data, train_labels,
              epochs=50,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model1.save_weights('bottleneck_fc_model.h5')'''








image_size = (FACE_WIDTH, FACE_HEIGHT)

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    #x = inputs
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (FACE_CHANNELS,), num_classes=num_classes)
#keras.utils.plot_model(model, show_shapes=True)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)





'''# Splitting the dataset for training and testing.

def is_test(x, _):
    return x % 20 == 0

def is_val(x, _):
    return (not is_test(x)) and (x % 10 == 0)

def is_train(x, y):
    return (not is_val(x, y)) and (not is_test(x, y))

recover = lambda x, y: y

# Split the dataset for training.
x_val = list(x_train).enumerate() \
    .filter(is_val) \
    .map(recover)

# Split the dataset for testing/validation.
x_test = list(x_train).enumerate() \
    .filter(is_test) \
    .map(recover)

# Split the dataset for testing/validation.
x_train = list(x_train).enumerate() \
    .filter(is_train) \
    .map(recover)'''

#callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5")]

'''tf.keras.callbacks.ModelCheckpoint(
    filepath,
    monitor="val_loss",
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    options=None,
    **kwargs
)'''

#time_stamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S_%f")
time_stamp = datetime.now().strftime("%Y_%m_%d__%H_%M")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="output/mask_model_" + MODEL_NAME + "_best_val_loss_{val_loss:.3f}_val_accuracy_{val_accuracy:.3f}_epoch_{epoch:03d}_loss_{loss:.3f}_accuracy_{accuracy:.3f}_" + str(FACE_WIDTH) + '_' + str(FACE_HEIGHT) + '_' + str(FACE_CHANNELS) + '__' + time_stamp + ".h5",
        save_weights_only=False,
        monitor='val_loss',
        mode='auto',  #'max',
        save_best_only=True),
    keras.callbacks.ModelCheckpoint(
        filepath="output/mask_model_" + MODEL_NAME + "_best_val_accuracy_{val_accuracy:.3f}_val_loss_{val_loss:.3f}_epoch_{epoch:03d}_loss_{loss:.3f}_accuracy_{accuracy:.3f}_" + str(FACE_WIDTH) + '_' + str(FACE_HEIGHT) + '_' + str(FACE_CHANNELS) + '__' + time_stamp + ".h5",
        save_weights_only=False,
        monitor='val_accuracy',
        mode='auto',
        save_best_only=True)
]

#datagen.fit(x_train)
#datagen_flow = datagen.flow(x_train, y_train, batch_size=32)

#model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds)
history = model.fit(
    #datagen_flow,
    #datagen.flow(x_train, y_train, batch_size=32),
    #x=x_train[:-1], #  list(np.moveaxis(x_train, -1, 0)),
    #y=y_train[:-1],
    x=x_train,
    y=y_train,
    batch_size=32,
    epochs=EPOCHS,
    #validation_data=(list(np.moveaxis(x_val, -1, 0)), y_val),
    validation_data=(x_val, y_val),
    #verbose=1,
    shuffle=True,
    callbacks=callbacks,
    #validation_split=0.2,
)

model.save_weights('output/result_mask_model_' + MODEL_NAME + '_' + str(FACE_WIDTH) + '_' + str(FACE_HEIGHT) + '_' + str(FACE_CHANNELS) + '__' + time_stamp + '.h5')

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

#results = model.evaluate(list(np.moveaxis(x_test, -1, 0)), y_test, batch_size=1)
#results = model.evaluate(list(np.moveaxis(x_val, -1, 0)), y_val, batch_size=1)

#results = model.evaluate(x_val, y_val, batch_size=1)
#print("test loss, test acc:", results)

#predictions = model.predict(list(np.moveaxis(x_test, -1, 0)))
#predictions = model.predict(list(np.moveaxis(x_val, -1, 0)))
#predictions = model.predict(x_train[-1].reshape((1, ) + x_train[-1].shape))
predictions = model.predict(x_test)
print("predictions shape:", predictions.shape)
predictions
#print("prediction [0, 0]:", predictions[0, 0], " should be:", y_test[0, 0, 0])
print("prediction [0, 0]:", predictions[0], " should be:", y_test[0])

input("Press Enter to continue...")

#img = keras.preprocessing.image.load_img("PetImages/Cat/6779.jpg", target_size=image_size)
#img_array = keras.preprocessing.image.img_to_array(img)
#img_array = tf.expand_dims(img_array, 0)  # Create batch axis
#predictions = model.predict(img_array)
#score = predictions[0]
#print("This image is %.2f percent cat and %.2f percent dog." % (100 * (1 - score), 100 * score))
