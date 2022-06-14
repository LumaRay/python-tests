import os
import cv2
import json
import shutil
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras

#OUTPUT_DATASET_NAME = "test1_maskfacesnewwork0312added1ffw1a1nb"
#MASK_MODEL_NAME = "mask_model_maskfacesnewwork12falsefaceswork1added1nb_best_val_accuracy_0.9037383198738098_val_loss_0.1538427323102951_epoch_42_loss_0.06606884300708771_accuracy_0.9512393474578857_2021_01_04__23_32_52_619227.h5"
#DATASET_PARSED_FOLDER = 'parsed_data_work0312_false_faces_work1_added1'
#CLASS_NAMES = ['mask', 'maskchin', 'masknone', 'masknose']

#OUTPUT_DATASET_NAME = "test1_maskfacesnewwork0312added1ffw1a1"
#MASK_MODEL_NAME = "mask_model_maskfacesnewwork0312added1ffw1a1_best_val_accuracy_0.947_val_loss_0.094_epoch_107_loss_0.017_accuracy_0.986_128_128_3__2021_01_05__01_30.h5"
#DATASET_PARSED_FOLDER = 'parsed_data_work0312_false_faces_work1_added1'
#CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']

#OUTPUT_DATASET_NAME = "test1_maskfacesnewwork12falsefaceswork1added1"
#MASK_MODEL_NAME = "mask_model_maskfacesnewwork0312added1ffw1a1_best_val_accuracy_0.947_val_loss_0.094_epoch_107_loss_0.017_accuracy_0.986_128_128_3__2021_01_05__01_30.h5"
#DATASET_PARSED_FOLDER = 'parsed_data_work12_false_faces_work1_added1'
#CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']

#OUTPUT_DATASET_NAME = "test1_vk_sources1exp1"
#MASK_MODEL_NAME = "mask_model_maskfacesnewwork0312added1ffw1a1_best_val_accuracy_0.947_val_loss_0.094_epoch_107_loss_0.017_accuracy_0.986_128_128_3__2021_01_05__01_30.h5"
#DATASET_PARSED_FOLDER = 'vk_sources1exp1'
#CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']

# OUTPUT_DATASET_NAME = "test1_vk_sources1exp2"
# MASK_MODEL_NAME = "mask_model_maskfacesnewwork0312added1ffw1a1vk1exp1_best_val_accuracy_0.951_val_loss_0.090_epoch_114_loss_0.025_accuracy_0.978_128_128_3__2021_01_09__03_59.h5"
# DATASET_PARSED_FOLDER = 'vk_sources1exp2'
# CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']

# OUTPUT_DATASET_NAME = "test2_worktestset_using_facemasknoses"
# MASK_MODEL_NAME = "good/mask_model_maskfacesnewwork12ffw1a1_effect__best_val_accuracy_0.944_val_loss_0.110_epoch_163_loss_0.014_accuracy_0.987_128_128_3__2021_01_16__14_59.h5"
# DATASET_PARSED_FOLDER = 'worktestset_using_facemasknoses'
# CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']

# OUTPUT_DATASET_NAME = "worktestsetadd1_using_facemasknoses_mask_model_sort"
# MASK_MODEL_NAME = "good/mask_model_maskfacesnewwork12ffw1a1_effect__best_val_accuracy_0.944_val_loss_0.110_epoch_163_loss_0.014_accuracy_0.987_128_128_3__2021_01_16__14_59.h5"
# DATASET_PARSED_FOLDER = 'worktestsetadd1_using_facemasknoses'
# CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']

OUTPUT_DATASET_NAME = "worktestset_using_facemasknoses_mask_model_sort"
MASK_MODEL_NAME = "good/mask_model_maskfacesnewwork12ffw1a1_effect__best_val_accuracy_0.944_val_loss_0.110_epoch_163_loss_0.014_accuracy_0.987_128_128_3__2021_01_16__14_59.h5"
DATASET_PARSED_FOLDER = 'worktestset_using_facemasknoses'
CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']

# OUTPUT_DATASET_NAME = "worktestsetadd1_using_maskfaces_mask_model_sort"
# MASK_MODEL_NAME = "good/mask_model_maskfacesnewwork12ffw1a1_effect__best_val_accuracy_0.944_val_loss_0.110_epoch_163_loss_0.014_accuracy_0.987_128_128_3__2021_01_16__14_59.h5"
# DATASET_PARSED_FOLDER = 'worktestsetadd1_using_maskfaces'
# CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']

# Model sort test stats:                                    bad     back    mask    maskchin    masknone    masknose    total
# worktestsetadd1_using_facemasknoses_mask_model_sort       122     9       164     4           11          100         410
# worktestsetadd1_using_facemasknoses_mask_nomask_sort      139     6       155     6           12          94          412
# worktestsetadd1_using_maskfaces_mask_model_sort           246     65      102     5           8           47          473
# worktestsetadd1_using_maskfaces_mask_nomask_sort          254     63      96      5           9           52          479

CHECK_OBJECT_IMAGES_EXIST = True

FACE_WIDTH = 128
FACE_HEIGHT = 128

MINIMUM_ACCURACY = 0.5

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

pathToScriptFolder = str(pathlib.Path().absolute().as_posix())

mask_model_path = pathToScriptFolder + '/../predict_mask/output/' + MASK_MODEL_NAME

parsed_data_path = pathToScriptFolder + '/../parsed_data/' + DATASET_PARSED_FOLDER
objects_source_path = parsed_data_path + '/objects'
images_source_path = parsed_data_path + '/frames'

dataset_sources_path = pathToScriptFolder + '/../dataset_sources'

classes_source_path = dataset_sources_path + '/' + OUTPUT_DATASET_NAME

mask_model = keras.models.load_model(mask_model_path)

object_files_list = os.listdir(objects_source_path)
for fidx, file in enumerate(object_files_list):
    if file.endswith(".txt"):
        file_name, file_ext = file.split('.')
        print("Parsing file " + str(fidx) + " of " + str(len(object_files_list)))
        object_image_found = os.path.isfile(objects_source_path + "/" + file_name + ".jpg")
        if CHECK_OBJECT_IMAGES_EXIST and (not object_image_found):
            continue
        with open(objects_source_path + '/' + file) as json_file:
            entry = json.load(json_file)
            frame_ref = entry['frame_ref']
            frame_width = entry['frame_width']
            frame_height = entry['frame_height']
            contour_area = entry['contour_area']
            bbox = entry['bbox']
            bbox_x = bbox['x']
            bbox_y = bbox['y']
            bbox_w = bbox['w']
            bbox_h = bbox['h']
            contour = entry['contour']
            np_contour = np.zeros((len(contour), 2), dtype=np.int32)
            for i, contour_point in enumerate(contour):
                x = contour_point['x'] - bbox_x
                y = contour_point['y'] - bbox_y
                np_contour[i] = [x, y]
            try:
                img = cv2.imread(images_source_path + '/' + frame_ref + ".jpg")
                img_roi = img[bbox_y:bbox_y + bbox_h, bbox_x:bbox_x + bbox_w]
                mask_roi = np.zeros(img_roi.shape[:2], np.uint8)
                cv2.drawContours(mask_roi, [np_contour], -1, (255), thickness=-1)
                res_roi = cv2.bitwise_and(img_roi, img_roi, mask=mask_roi)
                object_image = res_roi
                mask_roi -= 255
                gray_roi = np.zeros(img_roi.shape, np.uint8)
                gray_roi.fill(127)
                gray_roi_reversed = cv2.bitwise_and(gray_roi, gray_roi, mask=mask_roi)
                res_roi = cv2.addWeighted(res_roi, 1, gray_roi_reversed, 1, 0)
                res_roi = cv2.resize(res_roi, (FACE_WIDTH, FACE_HEIGHT), cv2.INTER_CUBIC)
                res_roi = res_roi.reshape((1,) + res_roi.shape)
                #res_roi = res_roi.copy()
                predictions = mask_model.predict(res_roi)
                idx_max = np.argmax(predictions[0])
                prob_max = predictions[0][idx_max]
                class_name = CLASS_NAMES[idx_max]
                if prob_max < MINIMUM_ACCURACY:
                    continue
                #title = "class={0} idx_max={1} prob_max={2:.2f}".format(class_name, idx_max, prob_max)
                #print(title)
                #cv2.imshow("123", res_roi[0])
                #cv2.waitKey(1)
                class_folder_path = classes_source_path + "/" + class_name
                if not os.path.exists(class_folder_path):
                    os.makedirs(class_folder_path)
                if object_image_found:
                    shutil.copy(objects_source_path + "/" + file_name + ".jpg", class_folder_path)
                else:
                    cv2.imwrite(class_folder_path + "/" + file_name + ".jpg", object_image)
            except:
                pass
