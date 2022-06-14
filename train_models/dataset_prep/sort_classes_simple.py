import os
import cv2
import json
import shutil
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras

#MASK_MODEL_NAME = "mask_model_maskfacesnewwork12falsefaceswork1added1nb_best_val_accuracy_0.9037383198738098_val_loss_0.1538427323102951_epoch_42_loss_0.06606884300708771_accuracy_0.9512393474578857_2021_01_04__23_32_52_619227.h5"
#MASK_MODEL_NAME = "mask_model_maskfacesnewwork0312added1ffw1a1_best_val_accuracy_0.947_val_loss_0.094_epoch_107_loss_0.017_accuracy_0.986_128_128_3__2021_01_05__01_30.h5"
MASK_MODEL_NAME = "mask_model_maskfacesnewwork0312added1ffw1a1vk1exp1_best_val_accuracy_0.951_val_loss_0.090_epoch_114_loss_0.025_accuracy_0.978_128_128_3__2021_01_09__03_59.h5"
CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']


FACE_WIDTH = 128
FACE_HEIGHT = 128

MINIMUM_ACCURACY = 0.1

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

pathToScriptFolder = str(pathlib.Path().absolute().as_posix())

#input_folder = r"C:\Users\Jure\Downloads\RMFD\self-built-masked-face-recognition-dataset\AFDB_masked_face_dataset"
#output_folder = r"C:\Users\Jure\Downloads\RMFD-out"

input_folder = r"C:\Users\Jure\Downloads\RMFD\self-built-masked-face-recognition-dataset\AFDB_masked_face_dataset"
output_folder = r"C:\Users\Jure\Downloads\RMFD-out"

#mask_model_path = pathToScriptFolder + '/../predict_mask/' + MASK_MODEL_NAME
mask_model_path = pathToScriptFolder + '/../../../face_detection/uem_mask/' + MASK_MODEL_NAME

mask_model = keras.models.load_model(mask_model_path)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def parseFolder(in_folder, out_folder):
    #object_files_list = os.listdir(in_folder)
    for root, subdirs, files in os.walk(in_folder):
        #for fidx, folder in enumerate(subdirs):
        #    print("Parsing directory " + str(fidx) + " of " + str(len(subdirs)))
        #    parseFolder(folder, out_folder)
        for fidx, filename in enumerate(files):
            if filename.endswith(".jpg"):
                #file_name, file_ext = file.split('.')
                print("Parsing file " + str(fidx) + " of " + str(len(files)))
                try:
                    #file_path = in_folder + '/' + filename
                    file_path = os.path.join(root, filename)
                    res_roi = cv2.imread(file_path)
                    res_roi = cv2.resize(res_roi, (FACE_WIDTH, FACE_HEIGHT), cv2.INTER_CUBIC)
                    res_roi = res_roi.reshape((1,) + res_roi.shape)
                    predictions = mask_model.predict(res_roi)
                    idx_max = np.argmax(predictions[0])
                    prob_max = predictions[0][idx_max]
                    if prob_max < MINIMUM_ACCURACY:
                        continue
                    class_name = CLASS_NAMES[idx_max]
                    #class_folder_path = out_folder + "/" + class_name
                    class_folder_path = os.path.join(out_folder, class_name)
                    if not os.path.exists(class_folder_path):
                        os.makedirs(class_folder_path)
                    shutil.copy(file_path, class_folder_path)
                except:
                    pass

parseFolder(input_folder, output_folder)