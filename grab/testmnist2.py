# https://keras.io/examples/vision/mnist_convnet/
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

MODEL_TYPE_DEFAULT = "default"
MODEL_TYPE_LSTM = "lstm"
MODEL_TYPE = MODEL_TYPE_LSTM

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
    return model

if MODEL_TYPE == MODEL_TYPE_DEFAULT:
    model = make_model_default()
elif MODEL_TYPE == MODEL_TYPE_LSTM:
    input_shape = (1, 28, 28, 1)
    model = make_model_lstm()
    x_train = np.expand_dims(x_train, 1)

model.summary()

batch_size = 128
epochs = 5

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

