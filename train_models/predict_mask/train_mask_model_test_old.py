# https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_from_scratch.py
import gc
import os
import cv2
import json
import glob
import random
import pathlib
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from multiprocessing import Process
from multiprocessing.spawn import freeze_support
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, Activation, MaxPooling3D, Dense, Dropout, Reshape, LSTM, Flatten, BatchNormalization, Input, concatenate
# USE_PRECISION:int = 16
# from tests.train_models.predict_mask.train_mask_model import PORTION_TEST

USE_PRECISION:int = 32
# USE_PRECISION:int = 64

# BATCH_SIZE = 32
BATCH_SIZE = 64

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

DATASET_NAMES_TINY = [
    'source_2020_11_25_noms',  # 46 files
]

DATASET_NAMES_SMALL = [
    'source_2020_11_25_1000ms',  # 358 files
]

DATASET_NAMES_CUSTOM = [
    # 'maskfacesnew',  # not work 3368 files
    # 'maskfacesnew3',  # not work 3057 files
    'maskfacesnewwork1',  # 1230 files
    'maskfacesnewwork2',  # 1811 files
    # 'maskfacesnewwork_toadd1',  # not work 3163 files
    'false_faces_work1',  # 470 files
    'source_2020_11_25_1000ms',  # 358 files
    'source_2020_11_25_1000ms5max',  # 215 files
    'source_2020_11_25_200ms',  # 176 files
    'source_2020_11_25_3000ms',  # 49 files
    'source_2020_11_25_3000ms5max',  # 312 files
    'source_2020_11_25_300ms',  # 144 files
    'source_2020_11_25_noms',  # 46 files
    # 'vk_sources1',  # not work 9491 files
    # 'worktestset_using_facemasknoses_mask_model_sort',  # 766 files
]

DATASET_NAMES_WORK = [
    'maskfacesnewwork1',
    'maskfacesnewwork2',
    'false_faces_work1',
    'source_2020_11_25_1000ms',
    'source_2020_11_25_1000ms5max',
    'source_2020_11_25_200ms',
    'source_2020_11_25_3000ms',
    'source_2020_11_25_3000ms5max',
    'source_2020_11_25_300ms',
    'source_2020_11_25_noms',
    # 'worktestset_using_facemasknoses_mask_model_sort',  # 766 files
]

DATASET_NAMES_WORKv12 = [
    'maskfacesnewwork1',
    'maskfacesnewwork2',
    'false_faces_work1',
    'source_2020_11_25_1000ms',
    'source_2020_11_25_1000ms5max',
    'source_2020_11_25_200ms',
    'source_2020_11_25_3000ms',
    'source_2020_11_25_3000ms5max',
    'source_2020_11_25_300ms',
    'source_2020_11_25_noms',
    'v2_source_2020_11_17',
    'v2_source_2020_11_19',
    'v2_source_2020_11_25_1000ms',
    'v2_source_2020_11_25_1000ms5max',
    'v2_source_2020_11_25_200ms',
    'v2_source_2020_11_25_3000ms',
    'v2_source_2020_11_25_3000ms5max',
    'v2_source_2020_11_25_300ms',
    'v2_source_2020_11_25_noms',
    # 'worktestset_using_facemasknoses_mask_model_sort',  # 766 files
]

DATASET_NAMES_VK = [
    'vk_sources1',  # not work
]

DATASET_NAMES_YT = [
    'maskfacesnew',  # not work
    'maskfacesnew3',  # not work
    'maskfacesnewwork_toadd1',  # not work
]

DATASET_NAMES_ALL = [
    'maskfacesnew',  # not work
    'maskfacesnew3',  # not work
    'maskfacesnewwork1',
    'maskfacesnewwork2',
    'maskfacesnewwork_toadd1',  # not work
    'false_faces_work1',
    'source_2020_11_25_1000ms',
    'source_2020_11_25_1000ms5max',
    'source_2020_11_25_200ms',
    'source_2020_11_25_3000ms',
    'source_2020_11_25_3000ms5max',
    'source_2020_11_25_300ms',
    'source_2020_11_25_noms',
    'vk_sources1',  # not work
    # 'worktestset_using_facemasknoses_mask_model_sort',  # 766 files
]

DATASET_NAMES_WORKTESTSET = [
    'worktestset_using_facemasknoses_mask_model_sort',  # 766 files
]

DATASET_NAMES_DEFAULT = DATASET_NAMES_WORK
# DATASET_NAMES_DEFAULT = DATASET_NAMES_TINY
# DATASET_NAMES_DEFAULT = DATASET_NAMES_SMALL

# SUPER_TEST_IMAGES_COUNT = 766

CLASSES_MAP = []

CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']

# CLASS_NAMES = ['back', 'mask', 'masknone']

# CLASS_NAMES = ['mask', 'maskchin', 'masknone', 'masknose']
# CLASSES_MAP = [0, 2, 2, 2]  # mask/masknone classes

if len(CLASSES_MAP) == 0:
    CLASSES_MAP = [ci for ci, _ in enumerate(CLASS_NAMES)]
# num_classes = len(CLASS_NAMES)
class_indexes = list(set(CLASSES_MAP))
new_class_names = [CLASS_NAMES[ci] for ci in class_indexes]
num_classes = len(new_class_names)


# MODEL_NAME = 'maskfacesnewwork0312added1ffw1a1vk1exp1'
# MODEL_NAME = 'maskfacesnewwork12ffw1a1'
# MODEL_NAME = 'maskfacesnewwork12ffw1a1nbnc'
# MODEL_NAME = 'testnewtrain'
# MODEL_NAME = 'test'
MODEL_NAME = ''

#parsed_data_path = pathToScriptFolder + '/../parsed_data/' + DATASET_PARSED_FOLDER + '/'
#objects_source_path = parsed_data_path + 'objects'
#images_source_path = parsed_data_path + 'frames'
parsed_data_path = pathToScriptFolder + '/../parsed_data'

#FACE_WIDTH = 64
#FACE_HEIGHT = 64
FACE_WIDTH = 128
FACE_HEIGHT = 128

FACE_CHANNELS = 3

# PORTION_VAL = 30
PORTION_VAL = 20
# USE_WORKTESTSET = False
USE_WORKTESTSET = True
if USE_WORKTESTSET:
    PORTION_TEST = 0
else:
    # PORTION_TEST = 1
    PORTION_TEST = 20

# EPOCHS = 1000
# EPOCHS = 200
# EPOCHS = 100
EPOCHS = 20
# EPOCHS = 1
RUNS_PER_EFFECT = 10
# RUNS_PER_EFFECT = 5
# RUNS_PER_EFFECT = 3
# RUNS_PER_EFFECT = 1
# EXT_EPOCHS = 0
# EXT_EPOCHS = 15
# EXT_EPOCHS = 100
EXT_EPOCHS = 200

# EXT_EPOCHS_BY_SIMPLE_ACC = False
# EXT_EPOCHS_BY_SIMPLE_ACC = True

MAX_CLASS_IMAGES = 999999
#MAX_CLASS_IMAGES = 100

SAVE_MODELS = True

# SAVE_BEST_LOSS_CHECKPOINTS = False
SAVE_BEST_LOSS_CHECKPOINTS = True
# SAVE_BEST_ACC_CHECKPOINTS = False
SAVE_BEST_ACC_CHECKPOINTS = True

# ADD_CHANNELS = False
# ADD_CHANNELS = True

# EXTEND_DATASET = False
# EXTEND_DATASET = True

#in_shape = (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS)

EFFECT_NONE = "or"
EFFECT_GRAY = "g"
EFFECT_HSV = "hs"
EFFECT_LAB = "la"
EFFECT_YCC = "yc"
EFFECT_BLUR = "bl"
EFFECT_SHARPEN = "sh"
EFFECT_BRIGHTEN = "br"
EFFECT_DARKEN = "dr"
EFFECT_CONTRAST_INC = "cn"
EFFECT_CONTRAST_DEC = "dc"
EFFECT_SATURATE = "sa"
EFFECT_DESATURATE = "ds"
EFFECT_ADAPTIVE = "ad"
EFFECT_NORMALIZE = "nr"
EFFECT_DEVIATION = "dv"
EFFECT_DEVIATION2 = "vv"
EFFECT_TEST1 = "t1"
EFFECT_TEST2 = "t2"
# EFFECT_ADD_DENSE = "dns"
# EFFECT_IN_SHAPE = "shp"
# EFFECT_PROPORTION = "prp"
# EFFECT_SCALE_FEATURES = "scl"
EFFECT_FLIP = "f"

ADD_CHANNELS_EFFECT_SEQUENCES_LIST = [
    [(EFFECT_NONE, "")],
    [(EFFECT_GRAY, 3)],
    # [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 6)],
    [(EFFECT_HSV, "")],
    [(EFFECT_YCC, "")],
    [(EFFECT_LAB, 3)],
    [(EFFECT_TEST1, "")],
    [(EFFECT_TEST2, "")],
    [(EFFECT_ADAPTIVE, 3)],
    [(EFFECT_BLUR, 9)],
    # [(EFFECT_BLUR, 3)],
    # [(EFFECT_BLUR, 1)],
    # [(EFFECT_SHARPEN, 1)],
    # [(EFFECT_SHARPEN, 3)],
    # [(EFFECT_SHARPEN, 7)],
    [(EFFECT_SHARPEN, 9)],
    # [(EFFECT_SHARPEN, 11)],
    # [(EFFECT_SHARPEN, 13)],
    [(EFFECT_BRIGHTEN, 10)],
    [(EFFECT_DARKEN, 10)],
    [(EFFECT_CONTRAST_INC, 10)],
    # [(EFFECT_CONTRAST_DEC, 6)],
    # [(EFFECT_CONTRAST_DEC, 8)],
    [(EFFECT_CONTRAST_DEC, 10)],
    # [(EFFECT_CONTRAST_DEC, 12)],
    # [(EFFECT_CONTRAST_DEC, 14)],
    [(EFFECT_SATURATE, 10)],
    # [(EFFECT_DESATURATE, 6)],
    # [(EFFECT_DESATURATE, 8)],
    [(EFFECT_DESATURATE, 10)],
    # [(EFFECT_DESATURATE, 12)],
    # [(EFFECT_DESATURATE, 14)],
    [(EFFECT_NORMALIZE, "")],
    [(EFFECT_DEVIATION, "")],
    [(EFFECT_DEVIATION2, "")],
]

ADD_CHANNELS_EFFECT_SEQUENCES_SHORT_LIST = [
    [(EFFECT_NONE, "")],
    [(EFFECT_GRAY, 3)],
    [(EFFECT_HSV, "")],
    [(EFFECT_YCC, "")],
    [(EFFECT_LAB, 3)],
    [(EFFECT_TEST1, "")],
    [(EFFECT_TEST2, "")],
    [(EFFECT_ADAPTIVE, 3)],
]

ADD_CHANNELS_EFFECT_SEQUENCES_GRAY_LIST = [
    # [(EFFECT_NONE, "")],
    [(EFFECT_GRAY, 3)],
    # [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 6)],
    [(EFFECT_GRAY, 3), (EFFECT_HSV, "")],
    [(EFFECT_GRAY, 3), (EFFECT_YCC, "")],
    [(EFFECT_GRAY, 3), (EFFECT_LAB, 3)],
    [(EFFECT_GRAY, 3), (EFFECT_TEST1, "")],
    [(EFFECT_GRAY, 3), (EFFECT_TEST2, "")],
    [(EFFECT_GRAY, 3), (EFFECT_ADAPTIVE, 3)],
    [(EFFECT_GRAY, 3), (EFFECT_BLUR, 9)],
    # [(EFFECT_BLUR, 3)],
    # [(EFFECT_BLUR, 1)],
    # [(EFFECT_SHARPEN, 1)],
    # [(EFFECT_SHARPEN, 3)],
    # [(EFFECT_SHARPEN, 7)],
    [(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9)],
    # [(EFFECT_SHARPEN, 11)],
    # [(EFFECT_SHARPEN, 13)],
    [(EFFECT_GRAY, 3), (EFFECT_BRIGHTEN, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_DARKEN, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_INC, 10)],
    # [(EFFECT_CONTRAST_DEC, 6)],
    # [(EFFECT_CONTRAST_DEC, 8)],
    [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 10)],
    # [(EFFECT_CONTRAST_DEC, 12)],
    # [(EFFECT_CONTRAST_DEC, 14)],
    [(EFFECT_GRAY, 3), (EFFECT_SATURATE, 10)],
    # [(EFFECT_DESATURATE, 6)],
    # [(EFFECT_DESATURATE, 8)],
    [(EFFECT_GRAY, 3), (EFFECT_DESATURATE, 10)],
    # [(EFFECT_DESATURATE, 12)],
    # [(EFFECT_DESATURATE, 14)],
    [(EFFECT_GRAY, 3), (EFFECT_NORMALIZE, "")],
    [(EFFECT_GRAY, 3), (EFFECT_DEVIATION, "")],
    [(EFFECT_GRAY, 3), (EFFECT_DEVIATION2, "")],
]

ADD_CHANNELS_EFFECT_SEQUENCES_GRAY_SHORT_LIST = [
    [(EFFECT_GRAY, 3)],
    [(EFFECT_GRAY, 3), (EFFECT_HSV, "")],
    [(EFFECT_GRAY, 3), (EFFECT_YCC, "")],
    [(EFFECT_GRAY, 3), (EFFECT_LAB, 3)],
    [(EFFECT_GRAY, 3), (EFFECT_TEST1, "")],
    [(EFFECT_GRAY, 3), (EFFECT_TEST2, "")],
    [(EFFECT_GRAY, 3), (EFFECT_ADAPTIVE, 3)],
]

EXTEND_DATASET_EFFECT_SEQUENCES_LIST = [
    # [[(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 6)]],
    [[(EFFECT_GRAY, 3)]],
    # [[(EFFECT_CONTRAST_DEC, 6)]],
    [[(EFFECT_BLUR, 3)]],
    # [[(EFFECT_BLUR, 1)]],
    # [[(EFFECT_SHARPEN, 1)]],
    # [[(EFFECT_SHARPEN, 3)]],
    # [[(EFFECT_SHARPEN, 7)]],
    [[(EFFECT_SHARPEN, 9)]],
    # [[(EFFECT_SHARPEN, 11)]],
    # [[(EFFECT_SHARPEN, 13)]],
    [[(EFFECT_BRIGHTEN, 10)]],
    [[(EFFECT_DARKEN, 10)]],
    [[(EFFECT_CONTRAST_INC, 10)]],
    # [[(EFFECT_CONTRAST_DEC, 6)]],
    # [[(EFFECT_CONTRAST_DEC, 8)]],
    [[(EFFECT_CONTRAST_DEC, 10)]],
    # [[(EFFECT_CONTRAST_DEC, 12)]],
    # [[(EFFECT_CONTRAST_DEC, 14)]],
    [[(EFFECT_NORMALIZE, "")]],
    [[(EFFECT_DEVIATION, "")]],
    [[(EFFECT_DEVIATION2, "")]],
]

EXTEND_DATASET_EFFECT_SEQUENCES_FLIP_LIST = [
    # [[(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 6)]],
    [[(EFFECT_GRAY, 3)]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, "")]],
    # [[(EFFECT_CONTRAST_DEC, 6)]],
    [[(EFFECT_BLUR, 3)]],
    [[(EFFECT_BLUR, 3), (EFFECT_FLIP, "")]],
    # [[(EFFECT_BLUR, 1)]],
    # [[(EFFECT_SHARPEN, 1)]],
    # [[(EFFECT_SHARPEN, 3)]],
    # [[(EFFECT_SHARPEN, 7)]],
    [[(EFFECT_SHARPEN, 9)]],
    [[(EFFECT_SHARPEN, 9), (EFFECT_FLIP, "")]],
    # [[(EFFECT_SHARPEN, 11)]],
    # [[(EFFECT_SHARPEN, 13)]],
    [[(EFFECT_BRIGHTEN, 10)]],
    [[(EFFECT_BRIGHTEN, 10), (EFFECT_FLIP, "")]],
    [[(EFFECT_DARKEN, 10)]],
    [[(EFFECT_DARKEN, 10), (EFFECT_FLIP, "")]],
    [[(EFFECT_CONTRAST_INC, 10)]],
    [[(EFFECT_CONTRAST_INC, 10), (EFFECT_FLIP, "")]],
    # [[(EFFECT_CONTRAST_DEC, 6)]],
    # [[(EFFECT_CONTRAST_DEC, 8)]],
    [[(EFFECT_CONTRAST_DEC, 10)]],
    [[(EFFECT_CONTRAST_DEC, 10), (EFFECT_FLIP, "")]],
    # [[(EFFECT_CONTRAST_DEC, 12)]],
    # [[(EFFECT_CONTRAST_DEC, 14)]],
    [[(EFFECT_NORMALIZE, "")]],
    [[(EFFECT_NORMALIZE, ""), (EFFECT_FLIP, "")]],
    [[(EFFECT_DEVIATION, "")]],
    [[(EFFECT_DEVIATION, ""), (EFFECT_FLIP, "")]],
    [[(EFFECT_DEVIATION2, "")]],
    [[(EFFECT_DEVIATION2, ""), (EFFECT_FLIP, "")]],
]

EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_LIST = [
    [[(EFFECT_NONE, "")]],
    [[(EFFECT_TEST1, "")]],
    [[(EFFECT_GRAY, 3)]],
    # [[(EFFECT_CONTRAST_DEC, 6)]],
    [[(EFFECT_BLUR, 3)]],
    # [[(EFFECT_BLUR, 1)]],
    # [[(EFFECT_SHARPEN, 1)]],
    # [[(EFFECT_SHARPEN, 3)]],
    # [[(EFFECT_SHARPEN, 7)]],
    [[(EFFECT_SHARPEN, 9)]],
    # [[(EFFECT_SHARPEN, 11)]],
    # [[(EFFECT_SHARPEN, 13)]],
    [[(EFFECT_BRIGHTEN, 10)]],
    [[(EFFECT_DARKEN, 10)]],
    [[(EFFECT_CONTRAST_INC, 10)]],
    # [[(EFFECT_CONTRAST_DEC, 6)]],
    # [[(EFFECT_CONTRAST_DEC, 8)]],
    [[(EFFECT_CONTRAST_DEC, 10)]],
    # [[(EFFECT_CONTRAST_DEC, 12)]],
    # [[(EFFECT_CONTRAST_DEC, 14)]],
    [[(EFFECT_NORMALIZE, "")]],
    [[(EFFECT_DEVIATION, "")]],
    [[(EFFECT_DEVIATION2, "")]],
]

EXTEND_DATASET_EFFECT_SEQUENCES_GRAY_LIST = [
    # [[(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 6)]],
    [[(EFFECT_GRAY, 3)]],
    # [[(EFFECT_CONTRAST_DEC, 6)]],
    [[(EFFECT_GRAY, 3), (EFFECT_BLUR, 3)]],
    # [[(EFFECT_BLUR, 1)]],
    # [[(EFFECT_SHARPEN, 1)]],
    # [[(EFFECT_SHARPEN, 3)]],
    # [[(EFFECT_SHARPEN, 7)]],
    [[(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9)]],
    # [[(EFFECT_SHARPEN, 11)]],
    # [[(EFFECT_SHARPEN, 13)]],
    [[(EFFECT_GRAY, 3), (EFFECT_BRIGHTEN, 10)]],
    [[(EFFECT_GRAY, 3), (EFFECT_DARKEN, 10)]],
    [[(EFFECT_GRAY, 3), (EFFECT_CONTRAST_INC, 10)]],
    # [[(EFFECT_CONTRAST_DEC, 6)]],
    # [[(EFFECT_CONTRAST_DEC, 8)]],
    [[(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 10)]],
    # [[(EFFECT_CONTRAST_DEC, 12)]],
    # [[(EFFECT_CONTRAST_DEC, 14)]],
    [[(EFFECT_GRAY, 3), (EFFECT_NORMALIZE, "")]],
    [[(EFFECT_GRAY, 3), (EFFECT_DEVIATION, "")]],
    [[(EFFECT_GRAY, 3), (EFFECT_DEVIATION2, "")]],
]

EXTEND_DATASET_EFFECT_SEQUENCES_GRAY_WITH_ORIG_TEST1_FLIP_LIST = [
    # [[(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 6)]],
    [[(EFFECT_NONE, 3)]],
    [[(EFFECT_FLIP, "")]],
    [[(EFFECT_TEST1, 3)]],
    [[(EFFECT_FLIP, ""), (EFFECT_TEST1, 3)]],
    [[(EFFECT_GRAY, 3)]],
    [[(EFFECT_FLIP, ""), (EFFECT_GRAY, 3)]],
    # [[(EFFECT_CONTRAST_DEC, 6)]],
    [[(EFFECT_GRAY, 3), (EFFECT_BLUR, 3)]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_BLUR, 3)]],
    # [[(EFFECT_BLUR, 1)]],
    # [[(EFFECT_SHARPEN, 1)]],
    # [[(EFFECT_SHARPEN, 3)]],
    # [[(EFFECT_SHARPEN, 7)]],
    [[(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9)]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_SHARPEN, 9)]],
    # [[(EFFECT_SHARPEN, 11)]],
    # [[(EFFECT_SHARPEN, 13)]],
    [[(EFFECT_GRAY, 3), (EFFECT_BRIGHTEN, 10)]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_BRIGHTEN, 10)]],
    [[(EFFECT_GRAY, 3), (EFFECT_DARKEN, 10)]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DARKEN, 10)]],
    [[(EFFECT_GRAY, 3), (EFFECT_CONTRAST_INC, 10)]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_INC, 10)]],
    # [[(EFFECT_CONTRAST_DEC, 6)]],
    # [[(EFFECT_CONTRAST_DEC, 8)]],
    [[(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 10)]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_DEC, 10)]],
    # [[(EFFECT_CONTRAST_DEC, 12)]],
    # [[(EFFECT_CONTRAST_DEC, 14)]],
    [[(EFFECT_GRAY, 3), (EFFECT_NORMALIZE, "")]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_NORMALIZE, "")]],
    [[(EFFECT_GRAY, 3), (EFFECT_DEVIATION, "")]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION, "")]],
    [[(EFFECT_GRAY, 3), (EFFECT_DEVIATION2, "")]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION2, "")]],
]

EXTEND_DATASET_EFFECT_SEQUENCES_GRAY_FLIP_LIST = [
    # [[(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 6)]],
    [[(EFFECT_GRAY, 3)]],
    [[(EFFECT_FLIP, ""), (EFFECT_GRAY, 3)]],
    # [[(EFFECT_CONTRAST_DEC, 6)]],
    [[(EFFECT_GRAY, 3), (EFFECT_BLUR, 3)]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_BLUR, 3)]],
    # [[(EFFECT_BLUR, 1)]],
    # [[(EFFECT_SHARPEN, 1)]],
    # [[(EFFECT_SHARPEN, 3)]],
    # [[(EFFECT_SHARPEN, 7)]],
    [[(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9)]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_SHARPEN, 9)]],
    # [[(EFFECT_SHARPEN, 11)]],
    # [[(EFFECT_SHARPEN, 13)]],
    [[(EFFECT_GRAY, 3), (EFFECT_BRIGHTEN, 10)]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_BRIGHTEN, 10)]],
    [[(EFFECT_GRAY, 3), (EFFECT_DARKEN, 10)]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DARKEN, 10)]],
    [[(EFFECT_GRAY, 3), (EFFECT_CONTRAST_INC, 10)]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_INC, 10)]],
    # [[(EFFECT_CONTRAST_DEC, 6)]],
    # [[(EFFECT_CONTRAST_DEC, 8)]],
    [[(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 10)]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_DEC, 10)]],
    # [[(EFFECT_CONTRAST_DEC, 12)]],
    # [[(EFFECT_CONTRAST_DEC, 14)]],
    [[(EFFECT_GRAY, 3), (EFFECT_NORMALIZE, "")]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_NORMALIZE, "")]],
    [[(EFFECT_GRAY, 3), (EFFECT_DEVIATION, "")]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION, "")]],
    [[(EFFECT_GRAY, 3), (EFFECT_DEVIATION2, "")]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION2, "")]],
]

EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_LIST = [
    [
        [(EFFECT_NONE, "")],
        [(EFFECT_GRAY, 3)],
        [(EFFECT_HSV, "")],
        [(EFFECT_YCC, "")],
        [(EFFECT_LAB, 3)],
        [(EFFECT_TEST1, "")],
        [(EFFECT_TEST2, "")],
        [(EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_SHARPEN, 9)],
        [(EFFECT_SHARPEN, 9), (EFFECT_GRAY, 3)],
        [(EFFECT_SHARPEN, 9), (EFFECT_HSV, "")],
        [(EFFECT_SHARPEN, 9), (EFFECT_YCC, "")],
        [(EFFECT_SHARPEN, 9), (EFFECT_LAB, 3)],
        [(EFFECT_SHARPEN, 9), (EFFECT_TEST1, "")],
        [(EFFECT_SHARPEN, 9), (EFFECT_TEST2, "")],
        [(EFFECT_SHARPEN, 9), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_BRIGHTEN, 10)],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_GRAY, 3)],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_HSV, "")],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_YCC, "")],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_LAB, 3)],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_TEST1, "")],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_TEST2, "")],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_DARKEN, 10)],
        [(EFFECT_DARKEN, 10), (EFFECT_GRAY, 3)],
        [(EFFECT_DARKEN, 10), (EFFECT_HSV, "")],
        [(EFFECT_DARKEN, 10), (EFFECT_YCC, "")],
        [(EFFECT_DARKEN, 10), (EFFECT_LAB, 3)],
        [(EFFECT_DARKEN, 10), (EFFECT_TEST1, "")],
        [(EFFECT_DARKEN, 10), (EFFECT_TEST2, "")],
        [(EFFECT_DARKEN, 10), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_CONTRAST_INC, 10)],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_GRAY, 3)],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_HSV, "")],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_YCC, "")],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_LAB, 3)],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_TEST1, "")],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_TEST2, "")],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_CONTRAST_DEC, 10)],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_GRAY, 3)],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_HSV, "")],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_YCC, "")],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_LAB, 3)],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_TEST1, "")],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_TEST2, "")],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_NORMALIZE, "")],
        [(EFFECT_NORMALIZE, ""), (EFFECT_GRAY, 3)],
        [(EFFECT_NORMALIZE, ""), (EFFECT_HSV, "")],
        [(EFFECT_NORMALIZE, ""), (EFFECT_YCC, "")],
        [(EFFECT_NORMALIZE, ""), (EFFECT_LAB, 3)],
        [(EFFECT_NORMALIZE, ""), (EFFECT_TEST1, "")],
        [(EFFECT_NORMALIZE, ""), (EFFECT_TEST2, "")],
        [(EFFECT_NORMALIZE, ""), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_DEVIATION, "")],
        [(EFFECT_DEVIATION, ""), (EFFECT_GRAY, 3)],
        [(EFFECT_DEVIATION, ""), (EFFECT_HSV, "")],
        [(EFFECT_DEVIATION, ""), (EFFECT_YCC, "")],
        [(EFFECT_DEVIATION, ""), (EFFECT_LAB, 3)],
        [(EFFECT_DEVIATION, ""), (EFFECT_TEST1, "")],
        [(EFFECT_DEVIATION, ""), (EFFECT_TEST2, "")],
        [(EFFECT_DEVIATION, ""), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_DEVIATION2, "")],
        [(EFFECT_DEVIATION2, ""), (EFFECT_GRAY, 3)],
        [(EFFECT_DEVIATION2, ""), (EFFECT_HSV, "")],
        [(EFFECT_DEVIATION2, ""), (EFFECT_YCC, "")],
        [(EFFECT_DEVIATION2, ""), (EFFECT_LAB, 3)],
        [(EFFECT_DEVIATION2, ""), (EFFECT_TEST1, "")],
        [(EFFECT_DEVIATION2, ""), (EFFECT_TEST2, "")],
        [(EFFECT_DEVIATION2, ""), (EFFECT_ADAPTIVE, 3)],
    ],
]

EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_FLIP_LIST = [
    [
        [(EFFECT_NONE, "")],
        [(EFFECT_GRAY, 3)],
        [(EFFECT_HSV, "")],
        [(EFFECT_YCC, "")],
        [(EFFECT_LAB, 3)],
        [(EFFECT_TEST1, "")],
        [(EFFECT_TEST2, "")],
        [(EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_NONE, ""), (EFFECT_FLIP, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, "")],
        [(EFFECT_HSV, ""), (EFFECT_FLIP, "")],
        [(EFFECT_YCC, ""), (EFFECT_FLIP, "")],
        [(EFFECT_LAB, 3), (EFFECT_FLIP, "")],
        [(EFFECT_TEST1, ""), (EFFECT_FLIP, "")],
        [(EFFECT_TEST2, ""), (EFFECT_FLIP, "")],
        [(EFFECT_ADAPTIVE, 3), (EFFECT_FLIP, "")],
    ],
    [
        [(EFFECT_SHARPEN, 9)],
        [(EFFECT_SHARPEN, 9), (EFFECT_GRAY, 3)],
        [(EFFECT_SHARPEN, 9), (EFFECT_HSV, "")],
        [(EFFECT_SHARPEN, 9), (EFFECT_YCC, "")],
        [(EFFECT_SHARPEN, 9), (EFFECT_LAB, 3)],
        [(EFFECT_SHARPEN, 9), (EFFECT_TEST1, "")],
        [(EFFECT_SHARPEN, 9), (EFFECT_TEST2, "")],
        [(EFFECT_SHARPEN, 9), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_SHARPEN, 9), (EFFECT_FLIP, "")],
        [(EFFECT_SHARPEN, 9), (EFFECT_GRAY, 3), (EFFECT_FLIP, "")],
        [(EFFECT_SHARPEN, 9), (EFFECT_HSV, ""), (EFFECT_FLIP, "")],
        [(EFFECT_SHARPEN, 9), (EFFECT_YCC, ""), (EFFECT_FLIP, "")],
        [(EFFECT_SHARPEN, 9), (EFFECT_LAB, 3), (EFFECT_FLIP, "")],
        [(EFFECT_SHARPEN, 9), (EFFECT_TEST1, ""), (EFFECT_FLIP, "")],
        [(EFFECT_SHARPEN, 9), (EFFECT_TEST2, ""), (EFFECT_FLIP, "")],
        [(EFFECT_SHARPEN, 9), (EFFECT_ADAPTIVE, 3), (EFFECT_FLIP, "")],
    ],
    [
        [(EFFECT_BRIGHTEN, 10)],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_GRAY, 3)],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_HSV, "")],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_YCC, "")],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_LAB, 3)],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_TEST1, "")],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_TEST2, "")],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_BRIGHTEN, 10), (EFFECT_FLIP, "")],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_GRAY, 3), (EFFECT_FLIP, "")],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_HSV, ""), (EFFECT_FLIP, "")],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_YCC, ""), (EFFECT_FLIP, "")],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_LAB, 3), (EFFECT_FLIP, "")],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_TEST1, ""), (EFFECT_FLIP, "")],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_TEST2, ""), (EFFECT_FLIP, "")],
        [(EFFECT_BRIGHTEN, 10), (EFFECT_ADAPTIVE, 3), (EFFECT_FLIP, "")],
    ],
    [
        [(EFFECT_DARKEN, 10)],
        [(EFFECT_DARKEN, 10), (EFFECT_GRAY, 3)],
        [(EFFECT_DARKEN, 10), (EFFECT_HSV, "")],
        [(EFFECT_DARKEN, 10), (EFFECT_YCC, "")],
        [(EFFECT_DARKEN, 10), (EFFECT_LAB, 3)],
        [(EFFECT_DARKEN, 10), (EFFECT_TEST1, "")],
        [(EFFECT_DARKEN, 10), (EFFECT_TEST2, "")],
        [(EFFECT_DARKEN, 10), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_DARKEN, 10), (EFFECT_FLIP, "")],
        [(EFFECT_DARKEN, 10), (EFFECT_GRAY, 3), (EFFECT_FLIP, "")],
        [(EFFECT_DARKEN, 10), (EFFECT_HSV, ""), (EFFECT_FLIP, "")],
        [(EFFECT_DARKEN, 10), (EFFECT_YCC, ""), (EFFECT_FLIP, "")],
        [(EFFECT_DARKEN, 10), (EFFECT_LAB, 3), (EFFECT_FLIP, "")],
        [(EFFECT_DARKEN, 10), (EFFECT_TEST1, ""), (EFFECT_FLIP, "")],
        [(EFFECT_DARKEN, 10), (EFFECT_TEST2, ""), (EFFECT_FLIP, "")],
        [(EFFECT_DARKEN, 10), (EFFECT_ADAPTIVE, 3), (EFFECT_FLIP, "")],
    ],
    [
        [(EFFECT_CONTRAST_INC, 10)],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_GRAY, 3)],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_HSV, "")],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_YCC, "")],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_LAB, 3)],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_TEST1, "")],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_TEST2, "")],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_FLIP, "")],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_GRAY, 3), (EFFECT_FLIP, "")],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_HSV, ""), (EFFECT_FLIP, "")],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_YCC, ""), (EFFECT_FLIP, "")],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_LAB, 3), (EFFECT_FLIP, "")],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_TEST1, ""), (EFFECT_FLIP, "")],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_TEST2, ""), (EFFECT_FLIP, "")],
        [(EFFECT_CONTRAST_INC, 10), (EFFECT_ADAPTIVE, 3), (EFFECT_FLIP, "")],
    ],
    [
        [(EFFECT_CONTRAST_DEC, 10)],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_GRAY, 3)],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_HSV, "")],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_YCC, "")],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_LAB, 3)],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_TEST1, "")],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_TEST2, "")],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_FLIP, "")],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_GRAY, 3), (EFFECT_FLIP, "")],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_HSV, ""), (EFFECT_FLIP, "")],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_YCC, ""), (EFFECT_FLIP, "")],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_LAB, 3), (EFFECT_FLIP, "")],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_TEST1, ""), (EFFECT_FLIP, "")],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_TEST2, ""), (EFFECT_FLIP, "")],
        [(EFFECT_CONTRAST_DEC, 10), (EFFECT_ADAPTIVE, 3), (EFFECT_FLIP, "")],
    ],
    [
        [(EFFECT_NORMALIZE, "")],
        [(EFFECT_NORMALIZE, ""), (EFFECT_GRAY, 3)],
        [(EFFECT_NORMALIZE, ""), (EFFECT_HSV, "")],
        [(EFFECT_NORMALIZE, ""), (EFFECT_YCC, "")],
        [(EFFECT_NORMALIZE, ""), (EFFECT_LAB, 3)],
        [(EFFECT_NORMALIZE, ""), (EFFECT_TEST1, "")],
        [(EFFECT_NORMALIZE, ""), (EFFECT_TEST2, "")],
        [(EFFECT_NORMALIZE, ""), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_NORMALIZE, ""), (EFFECT_FLIP, "")],
        [(EFFECT_NORMALIZE, ""), (EFFECT_GRAY, 3), (EFFECT_FLIP, "")],
        [(EFFECT_NORMALIZE, ""), (EFFECT_HSV, ""), (EFFECT_FLIP, "")],
        [(EFFECT_NORMALIZE, ""), (EFFECT_YCC, ""), (EFFECT_FLIP, "")],
        [(EFFECT_NORMALIZE, ""), (EFFECT_LAB, 3), (EFFECT_FLIP, "")],
        [(EFFECT_NORMALIZE, ""), (EFFECT_TEST1, ""), (EFFECT_FLIP, "")],
        [(EFFECT_NORMALIZE, ""), (EFFECT_TEST2, ""), (EFFECT_FLIP, "")],
        [(EFFECT_NORMALIZE, ""), (EFFECT_ADAPTIVE, 3), (EFFECT_FLIP, "")],
    ],
    [
        [(EFFECT_DEVIATION, "")],
        [(EFFECT_DEVIATION, ""), (EFFECT_GRAY, 3)],
        [(EFFECT_DEVIATION, ""), (EFFECT_HSV, "")],
        [(EFFECT_DEVIATION, ""), (EFFECT_YCC, "")],
        [(EFFECT_DEVIATION, ""), (EFFECT_LAB, 3)],
        [(EFFECT_DEVIATION, ""), (EFFECT_TEST1, "")],
        [(EFFECT_DEVIATION, ""), (EFFECT_TEST2, "")],
        [(EFFECT_DEVIATION, ""), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_DEVIATION, ""), (EFFECT_FLIP, "")],
        [(EFFECT_DEVIATION, ""), (EFFECT_GRAY, 3), (EFFECT_FLIP, "")],
        [(EFFECT_DEVIATION, ""), (EFFECT_HSV, ""), (EFFECT_FLIP, "")],
        [(EFFECT_DEVIATION, ""), (EFFECT_YCC, ""), (EFFECT_FLIP, "")],
        [(EFFECT_DEVIATION, ""), (EFFECT_LAB, 3), (EFFECT_FLIP, "")],
        [(EFFECT_DEVIATION, ""), (EFFECT_TEST1, ""), (EFFECT_FLIP, "")],
        [(EFFECT_DEVIATION, ""), (EFFECT_TEST2, ""), (EFFECT_FLIP, "")],
        [(EFFECT_DEVIATION, ""), (EFFECT_ADAPTIVE, 3), (EFFECT_FLIP, "")],
    ],
    [
        [(EFFECT_DEVIATION2, "")],
        [(EFFECT_DEVIATION2, ""), (EFFECT_GRAY, 3)],
        [(EFFECT_DEVIATION2, ""), (EFFECT_HSV, "")],
        [(EFFECT_DEVIATION2, ""), (EFFECT_YCC, "")],
        [(EFFECT_DEVIATION2, ""), (EFFECT_LAB, 3)],
        [(EFFECT_DEVIATION2, ""), (EFFECT_TEST1, "")],
        [(EFFECT_DEVIATION2, ""), (EFFECT_TEST2, "")],
        [(EFFECT_DEVIATION2, ""), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_DEVIATION2, ""), (EFFECT_FLIP, "")],
        [(EFFECT_DEVIATION2, ""), (EFFECT_GRAY, 3), (EFFECT_FLIP, "")],
        [(EFFECT_DEVIATION2, ""), (EFFECT_HSV, ""), (EFFECT_FLIP, "")],
        [(EFFECT_DEVIATION2, ""), (EFFECT_YCC, ""), (EFFECT_FLIP, "")],
        [(EFFECT_DEVIATION2, ""), (EFFECT_LAB, 3), (EFFECT_FLIP, "")],
        [(EFFECT_DEVIATION2, ""), (EFFECT_TEST1, ""), (EFFECT_FLIP, "")],
        [(EFFECT_DEVIATION2, ""), (EFFECT_TEST2, ""), (EFFECT_FLIP, "")],
        [(EFFECT_DEVIATION2, ""), (EFFECT_ADAPTIVE, 3), (EFFECT_FLIP, "")],
    ],
]

EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_GRAY_LIST = [
    [
        [(EFFECT_GRAY, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9)],
        [(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_GRAY, 3), (EFFECT_BRIGHTEN, 10)],
        [(EFFECT_GRAY, 3), (EFFECT_BRIGHTEN, 10), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_BRIGHTEN, 10), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_BRIGHTEN, 10), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_BRIGHTEN, 10), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_BRIGHTEN, 10), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_BRIGHTEN, 10), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_GRAY, 3), (EFFECT_DARKEN, 10)],
        [(EFFECT_GRAY, 3), (EFFECT_DARKEN, 10), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_DARKEN, 10), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_DARKEN, 10), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_DARKEN, 10), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_DARKEN, 10), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_DARKEN, 10), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_INC, 10)],
        [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_INC, 10), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_INC, 10), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_INC, 10), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_INC, 10), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_INC, 10), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_INC, 10), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 10)],
        [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 10), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 10), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 10), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 10), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 10), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 10), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_GRAY, 3), (EFFECT_NORMALIZE, "")],
        [(EFFECT_GRAY, 3), (EFFECT_NORMALIZE, ""), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_NORMALIZE, ""), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_NORMALIZE, ""), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_NORMALIZE, ""), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_NORMALIZE, ""), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_NORMALIZE, ""), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_GRAY, 3), (EFFECT_DEVIATION, "")],
        [(EFFECT_GRAY, 3), (EFFECT_DEVIATION, ""), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_DEVIATION, ""), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_DEVIATION, ""), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_DEVIATION, ""), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_DEVIATION, ""), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_DEVIATION, ""), (EFFECT_ADAPTIVE, 3)],
    ],
    [
        [(EFFECT_GRAY, 3), (EFFECT_DEVIATION2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_DEVIATION2, ""), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_DEVIATION2, ""), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_DEVIATION2, ""), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_DEVIATION2, ""), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_DEVIATION2, ""), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_DEVIATION2, ""), (EFFECT_ADAPTIVE, 3)],
    ],
]

'''if ADD_CHANNELS:
    FACE_CHANNELS *= 1 + len(ADD_CHANNELS_EFFECT_SEQUENCES_LIST)'''

MODEL_TUNE_DEFAULT = {
    "scale_features": 1,
    "add_dense": [],
    "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS),
    "conv_pyramid": [128, 256, 512, 728, 1024],
}

TASK_OPTIONS_DEFAULT = {  #  220 total epochs
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "runs": 10,
    "epochs": 20,
    "ext_epochs": 200,
}

TASK_OPTIONS_EPOCHS_1x1600x0 = {  #  1600 total epochs
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "runs": 1,
    "epochs": 1600,
    "ext_epochs": 0,
}

TASK_OPTIONS_EPOCHS_x20x40x800 = {  #  x4 total increase 840 total epochs
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "runs": 10 * 2,
    "epochs": 20 * 2,
    "ext_epochs": 200 * 4,
}

TASK_OPTIONS_EPOCHS_x10x40x400 = {  #  x4 total increase 840 total epochs
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "runs": 10,
    "epochs": 20 * 2,
    "ext_epochs": 200 * 2,
}

TASK_OPTIONS_EPOCHS_x40x80x800 = {  #  x10 total increase 840 total epochs
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "runs": 10 * 4,
    "epochs": 20 * 4,
    "ext_epochs": 200 * 4,
}

TASK_OPTIONS_EPOCHS_x20x40x1600 = {  #  x6 total increase 1240 total epochs
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "runs": 10 * 2,
    "epochs": 20 * 2,
    "ext_epochs": 200 * 8,
}

TASK_OPTIONS_EPOCHS_x20x40x3200 = {  #  x6 total increase 2040 total epochs
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "runs": 10 * 2,
    "epochs": 20 * 2,
    "ext_epochs": 200 * 16,
}

TEST_TASKS_LIST = [
    # {"effects": [[[(EFFECT_GRAY, 3)], [(EFFECT_LAB, 3)], [(EFFECT_HSV, "")], [(EFFECT_YCC, "")], [(EFFECT_TEST1, "")], [(EFFECT_TEST2, "")]]], "model_tune": {"scale_features": 1, "add_dense": [], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [1024, 512, 256, 512, 728, 1024]}},
    # {"effects": [ADD_CHANNELS_EFFECT_SEQUENCES_GRAY_LIST]},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_GRAY_LIST},
    # {"effects": [ADD_CHANNELS_EFFECT_SEQUENCES_LIST]},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_LIST},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "model_tune": {"scale_features": 1, "add_dense": [], "in_shape": (1, int(FACE_WIDTH / 2), int(FACE_HEIGHT / 2), FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "model_tune": {"scale_features": 1, "add_dense": [], "in_shape": (1, int(FACE_WIDTH * 2), int(FACE_HEIGHT * 2), FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}},
    # {"effects": [(EFFECT_NONE, "")], "add_channels": ADD_CHANNELS_EFFECT_SEQUENCES_LIST},
    # {"effects": [(EFFECT_NONE, "")], "dataset_names": DATASET_NAMES_DEFAULT},
    # {"effects": [[[(EFFECT_NONE, "")]]], "dataset_names": DATASET_NAMES_VK, "sub_name": "vk"},
    # {"effects": [[[(EFFECT_NONE, "")]]], "dataset_names": DATASET_NAMES_WORK, "sub_name": "work"},
    # {"effects": [[[(EFFECT_NONE, "")]]], "dataset_names": DATASET_NAMES_YT, "sub_name": "yt"},
    # {"effects": [[[(EFFECT_NONE, "")]]], "dataset_names": DATASET_NAMES_ALL, "sub_name": "all"},
    # {"effects": [(EFFECT_NONE, "")]},
    # {"effects": [[[(EFFECT_NONE, "")]]]},
    # {"effects": [(EFFECT_GRAY, 3)]},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "model_tune": {"scale_features": 0.5, "add_dense": [], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "model_tune": {"scale_features": 2, "add_dense": [], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "model_tune": {"scale_features": 1, "add_dense": [], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024, 2048]}},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "model_tune": {"scale_features": 1, "add_dense": [], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [64, 128, 256, 512, 728, 1024]}},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "model_tune": {"scale_features": 1, "add_dense": [], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [1024, 512, 256, 512, 728, 1024]}},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "model_tune": {"scale_features": 1, "add_dense": [], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [256, 512, 728, 1024]}},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "model_tune": {"scale_features": 1, "add_dense": [], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728]}},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "model_tune": {"scale_features": 1, "add_dense": [512], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "model_tune": {"scale_features": 1, "add_dense": [1024], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "model_tune": {"scale_features": 1, "add_dense": [2048], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "model_tune": {"scale_features": 1, "add_dense": [512, 512], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "model_tune": {"scale_features": 1, "add_dense": [1024, 1024], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "options": {"valid_proportion": 10, "test_proportion": PORTION_TEST, "epochs": EPOCHS, "ext_epochs": EXT_EPOCHS, "runs": RUNS_PER_EFFECT}},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "options": {"valid_proportion": 30, "test_proportion": PORTION_TEST, "epochs": EPOCHS, "ext_epochs": EXT_EPOCHS, "runs": RUNS_PER_EFFECT}},
    # {"effects": [[[(EFFECT_GRAY, 3)]]]},
    # {"effects": [(EFFECT_NONE, "")], "extend_dataset": [[(EFFECT_FLIP, "")]]},
    # {"effects": [[[(EFFECT_NONE, "")]], [[(EFFECT_FLIP, "")]]]},
    # {"effects": [(EFFECT_GRAY, 3)], "extend_dataset": [[(EFFECT_FLIP, "")]]},
    # {"effects": [[[(EFFECT_GRAY, 3)]], [[(EFFECT_FLIP, "")]]]},
    # {"effects": [(EFFECT_NONE, "")], "model_tune": MODEL_TUNE_DEFAULT, "options": TASK_OPTIONS_DEFAULT},
    # {"effects": [(EFFECT_NONE, "")], "add_channels": [[(EFFECT_NONE, "")]]},
    # {"effects": [[[(EFFECT_GRAY, 3)], [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 10)]]]},
    # {"effects": [[[(EFFECT_GRAY, 3)], [(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9)]]]},
    # {"effects": [[[(EFFECT_GRAY, 3)], [(EFFECT_GRAY, 3), (EFFECT_BLUR, 9)]]]},
    # {"effects": [[[(EFFECT_GRAY, 3)], [(EFFECT_GRAY, 3), (EFFECT_BRIGHTEN, 10)]]]},
    # {"effects": [(EFFECT_NONE, "")], "add_channels": [[(EFFECT_GRAY, 3)]]},
    # {"effects": [(EFFECT_NONE, "")], "extend_dataset": EXTEND_DATASET_EFFECT_SEQUENCES_LIST},
    # {"effects": [(EFFECT_NONE, "")]},
    # {"effects": [(EFFECT_GRAY, 3)]},
    # {"effects": [(EFFECT_NONE, "")], "extend_dataset": [[(EFFECT_GRAY, 3)]]},
    # {"effect": EFFECT_SCALE_FEATURES, "value": 0.5},
    # {"effect": EFFECT_SCALE_FEATURES, "value": 2},
    # {"effect": EFFECT_PROPORTION, "value": 30},
    # {"effect": EFFECT_PROPORTION, "value": 40},
    # {"effect": EFFECT_ADD_DENSE, "value": {"count": 1, "size": 1024}},
    # {"effect": EFFECT_ADD_DENSE, "value": {"count": 1, "size": 2048}},
    # {"effect": EFFECT_ADD_DENSE, "value": {"count": 1, "size": 4096}},
    # {"effect": EFFECT_ADD_DENSE, "value": {"count": 2, "size": 1024}},
    # {"effect": EFFECT_ADD_DENSE, "value": {"count": 3, "size": 1024}},
    # {"effect": EFFECT_IN_SHAPE, "value": (1, 128, 128, 1)},
    # {"effect": EFFECT_IN_SHAPE, "value": (1, 256, 256, 3)},
    # {"effect": EFFECT_IN_SHAPE, "value": (1, 192, 192, 3)},
    # {"effect": EFFECT_IN_SHAPE, "value": (1, 96, 96, 3)},
    # {"effect": EFFECT_IN_SHAPE, "value": (1, 64, 64, 3)},
    # {"effects": [[[(EFFECT_GRAY, 3)]]]},
    # {"effects": [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 6)]},
    # {"effects": [(EFFECT_GRAY, 3)], "model_tune": {**MODEL_TUNE_DEFAULT, "conv_pyramid": [128, 256, 512, 1024, 2048, 4096]}},
    # {"effect": EFFECT_GRAY, "value": 0},
    # {"effects": [(EFFECT_HSV, "")]},
    # {"effects": [(EFFECT_YCC, "")]},
    # {"effects": [(EFFECT_LAB, 3)]},
    # {"effect": EFFECT_LAB, "value": 0},
    # {"effects": [(EFFECT_BLUR, 5)]},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_BLUR, 5)]]]},
    # {"effect": EFFECT_BLUR, "value": 3},
    # {"effect": EFFECT_BLUR, "value": 1},
    # {"effect": EFFECT_SHARPEN, "value": 1},
    # {"effect": EFFECT_SHARPEN, "value": 3},
    # {"effect": EFFECT_SHARPEN, "value": 7},
    # {"effects": [(EFFECT_SHARPEN, 9)]},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9)]]]},
    # {"effect": EFFECT_SHARPEN, "value": 11},
    # {"effect": EFFECT_SHARPEN, "value": 13},
    # {"effects": [(EFFECT_BRIGHTEN, 10)]},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_BRIGHTEN, 10)]]]},
    # {"effects": [(EFFECT_DARKEN, 10)]},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_DARKEN, 10)]]]},
    # {"effects": [(EFFECT_CONTRAST_INC, 10)]},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_CONTRAST_INC, 10)]]]},
    # {"effect": EFFECT_CONTRAST_DEC, "value": 6},
    # {"effect": EFFECT_CONTRAST_DEC, "value": 8},
    # {"effects": [(EFFECT_CONTRAST_DEC, 10)]},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 10)]]]},
    # {"effect": EFFECT_CONTRAST_DEC, "value": 12},
    # {"effect": EFFECT_CONTRAST_DEC, "value": 14},
    # {"effects": [(EFFECT_SATURATE, 10)]},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_SATURATE, 10)]]]},
    # {"effect": EFFECT_DESATURATE, "value": 6},
    # {"effect": EFFECT_DESATURATE, "value": 8},
    # {"effects": [(EFFECT_DESATURATE, 10)]},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_DESATURATE, 10)]]]},
    # {"effect": EFFECT_DESATURATE, "value": 12},
    # {"effect": EFFECT_DESATURATE, "value": 14},
    # {"effects": [(EFFECT_ADAPTIVE, 3)]},
    # {"effects": [(EFFECT_NORMALIZE, "")]},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_NORMALIZE, "")]]]},
    # {"effects": [(EFFECT_DEVIATION, "")]},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_DEVIATION, "")]]]},
    # {"effects": [(EFFECT_DEVIATION2, "")]},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_DEVIATION2, "")]]]},
    # {"effects": [(EFFECT_TEST1, "")]},
    # {"effects": [(EFFECT_TEST2, "")]},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "dataset_names": DATASET_NAMES_VK, "sub_name": "vk"},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "dataset_names": DATASET_NAMES_WORK, "sub_name": "work"},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_GRAY_FLIP_LIST},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "dataset_names": DATASET_NAMES_YT, "sub_name": "yt"},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "dataset_names": DATASET_NAMES_ALL, "sub_name": "all"},
    # {"effects": [[[(EFFECT_GRAY, 3)]], [[(EFFECT_GRAY, 3), (EFFECT_FLIP, "")]]]},
    # {"effects": [[[(EFFECT_NONE, "")]], [[(EFFECT_GRAY, 3)]]]},
    # {"effects": [[[(EFFECT_NONE, "")]], [[(EFFECT_FLIP, "")]], [[(EFFECT_GRAY, 3)]], [[(EFFECT_GRAY, 3), (EFFECT_FLIP, "")]]]},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_LIST, "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "options": TASK_OPTIONS_EPOCHS_x2x2x4, "model_tune": {"scale_features": 1, "add_dense": [], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728]}},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_NORMALIZE, "")]]], "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_DEVIATION2, "")]]], "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_DEVIATION, "")]]], "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_DARKEN, 10)]]], "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9)]]], "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_BRIGHTEN, 10)]]], "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 10)]]], "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": [[[(EFFECT_GRAY, 3)], [(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9)]]], "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": [[[(EFFECT_GRAY, 3)], [(EFFECT_GRAY, 3), (EFFECT_BRIGHTEN, 10)]]], "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": [ADD_CHANNELS_EFFECT_SEQUENCES_GRAY_LIST], "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "dataset_names": DATASET_NAMES_ALL, "sub_name": "all", "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": [ADD_CHANNELS_EFFECT_SEQUENCES_GRAY_LIST], "model_tune": {"scale_features": 2, "add_dense": [], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}},
    # {"effects": [ADD_CHANNELS_EFFECT_SEQUENCES_GRAY_LIST], "model_tune": {"scale_features": 1, "add_dense": [], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [1024, 256, 512, 728, 1024]}},
    # {"effects": [[[(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9)]]], "options": TASK_OPTIONS_EPOCHS_x2x2x8},
    # {"effects": [[[(EFFECT_GRAY, 3)]], [[(EFFECT_GRAY, 3), (EFFECT_FLIP, "")]]], "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": [[[(EFFECT_NONE, "")]], [[(EFFECT_FLIP, "")]], [[(EFFECT_GRAY, 3)]], [[(EFFECT_GRAY, 3), (EFFECT_FLIP, "")]]], "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "model_tune": {"scale_features": 1, "add_dense": [], "in_shape": (1, 64, 64, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "model_tune": {"scale_features": 1, "add_dense": [], "in_shape": (1, 32, 32, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_LIST, "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": [[[(EFFECT_GRAY, 3)]]], "model_tune": {"scale_features": 1, "add_dense": [], "in_shape": (1, 64, 64, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}, "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_GRAY_LIST, "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_LIST, "model_tune": {"scale_features": 2, "add_dense": [], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}, "sub_name": "extadd"},  #  insufficient RAM (43299, 128, 128, 24) = 17025859584
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_LIST, "model_tune": {"scale_features": 1, "add_dense": [], "in_shape": (1, 64, 64, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}, "options": TASK_OPTIONS_EPOCHS_x2x2x8},
    # {"effects": EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_GRAY_LIST, "model_tune": {"scale_features": 2, "add_dense": [], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}, "sub_name": "extaddgray"},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_LIST, "options": TASK_OPTIONS_EPOCHS_x4x4x4},
    # {"effects": [[[(EFFECT_GRAY, 3)]], [[(EFFECT_GRAY, 3)]]], "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_LIST, "dataset_names": DATASET_NAMES_ALL, "sub_name": "all"},  #  MemoryError: Unable to allocate 10.9 GiB for an array with shape (238900, 128, 128, 3) and data type uint8
    {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_LIST, "dataset_names": DATASET_NAMES_WORKv12, "sub_name": "work12"},
    # {"effects": EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_FLIP_LIST, "dataset_names": DATASET_NAMES_ALL, "model_tune": {"scale_features": 2, "add_dense": [], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}, "sub_name": "allextadd"},
    {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_LIST, "options": TASK_OPTIONS_EPOCHS_x20x40x3200},
    {"effects": EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_LIST, "model_tune": {"scale_features": 2, "add_dense": [], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}, "sub_name": "extadd1", "options": TASK_OPTIONS_EPOCHS_1x1600x0},
    {"effects": EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_GRAY_LIST, "model_tune": {"scale_features": 2, "add_dense": [], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}, "sub_name": "extaddgray1", "options": TASK_OPTIONS_EPOCHS_1x1600x0},
    {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_FLIP_LIST, "dataset_names": DATASET_NAMES_ALL, "model_tune": {"scale_features": 2, "add_dense": [], "in_shape": (1, 64, 64, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}, "sub_name": "all", "options": TASK_OPTIONS_EPOCHS_x10x40x400},
    {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_GRAY_FLIP_LIST, "dataset_names": DATASET_NAMES_ALL, "model_tune": {"scale_features": 2, "add_dense": [], "in_shape": (1, 64, 64, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}, "sub_name": "all", "options": TASK_OPTIONS_EPOCHS_x10x40x400},
    {"effects": EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_FLIP_LIST, "dataset_names": DATASET_NAMES_ALL, "model_tune": {"scale_features": 2, "add_dense": [], "in_shape": (1, 64, 64, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}, "sub_name": "allextaddflip64", "options": TASK_OPTIONS_EPOCHS_x10x40x400},
    {"effects": EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_GRAY_LIST, "dataset_names": DATASET_NAMES_ALL, "model_tune": {"scale_features": 2, "add_dense": [], "in_shape": (1, 64, 64, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}, "sub_name": "allextaddgray64", "options": TASK_OPTIONS_EPOCHS_x10x40x400},
    {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_FLIP_LIST, "dataset_names": DATASET_NAMES_ALL, "model_tune": {"scale_features": 2, "add_dense": [], "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}, "sub_name": "all", "options": TASK_OPTIONS_EPOCHS_x20x40x800},
]
# tensorflow.python.framework.errors_impl.ResourceExhaustedError:  OOM when allocating tensor with shape[32,1024,64,64] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc 	 [[node model_10/batch_normalization_113/FusedBatchNormV3 (defined at /Work/InfraredCamera/ThermalView/tests/train_models/predict_mask/train_mask_model_test_old.py:1488) ]] Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info. [Op:__inference_train_function_859884]
'''if EXTEND_DATASET:
    SUPER_TEST_IMAGES_COUNT *= 1 + len(EXTEND_DATASET_EFFECT_SEQUENCES_LIST)'''


'''TRAIN_FACES_COUNT = 500
VALID_FACES_COUNT = 60
TEST_FACES_COUNT = 11

x_train = np.zeros((TRAIN_FACES_COUNT, FACE_HEIGHT, FACE_WIDTH, FACE_CHANNELS), dtype=np.float64)
y_train = np.zeros((TRAIN_FACES_COUNT, NUM_CLASSES), dtype=np.float64)

x_val = np.zeros((VALID_FACES_COUNT, FACE_HEIGHT, FACE_WIDTH, FACE_CHANNELS), dtype=np.float64)
y_val = np.zeros((VALID_FACES_COUNT, NUM_CLASSES), dtype=np.float64)

x_test = np.zeros((TEST_FACES_COUNT, FACE_HEIGHT, FACE_WIDTH, FACE_CHANNELS), dtype=np.float64)
y_test = np.zeros((TEST_FACES_COUNT, NUM_CLASSES), dtype=np.float64)'''

# gpu_options = tf.GPUOptions(allow_growth=True)
# session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

#classes_source_path = dataset_sources_path + '/' + DATASET_NAME

#dataset_outputs_path = pathToScriptFolder + '/../dataset_outputs'
#train_output_path = dataset_outputs_path + '/' + DATASET_NAME + '/train/images'
#valid_output_path = dataset_outputs_path + '/' + DATASET_NAME + '/valid/images'

#temp_path = pathToScriptFolder + '/../temp'
#if not os.path.exists(temp_path):
#    os.makedirs(temp_path)

'''datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')'''

'''def AHE(img):
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq
datagen = ImageDataGenerator(rotation_range=30, horizontal_flip=0.5, preprocessing_function=AHE)'''

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

def applyEffectsList(img, effects_list):
    for effect_type, effect_value in effects_list:
        img = applyEffect(img, effect_type, effect_value)
    return img

def applyEffect(img, effect_type, effect_value):
    if effect_type == EFFECT_GRAY:
        if effect_value >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif effect_value >= 0:
            tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.zeros_like(img, dtype=np.uint8)
            img[:, :, effect_value] = tmp
        elif effect_value == -1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif effect_type == EFFECT_HSV:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif effect_type == EFFECT_LAB:
        if effect_value >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        elif effect_value >= 0:
            img_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            img = np.zeros_like(img)
            img[:, :, effect_value] = img_tmp[:, :, 0]
        elif effect_value == -1:
            img_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            img = img_tmp[:, :, 0]
    elif effect_type == EFFECT_YCC:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif effect_type == EFFECT_BLUR:
        img = cv2.GaussianBlur(img, (effect_value * 2 + 1, effect_value * 2 + 1), 0)
    elif effect_type == EFFECT_SHARPEN:
        img_blurred = cv2.GaussianBlur(img, (effect_value * 2 + 1, effect_value * 2 + 1), 0)
        img = cv2.addWeighted(img, 1.5, img_blurred, -0.5, 0)
    elif effect_type == EFFECT_BRIGHTEN:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) + ((255 - hsv[:, :, 2].astype(np.float32)) * (effect_value / 100)), 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif effect_type == EFFECT_DARKEN:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) - (hsv[:, :, 2].astype(np.float32) * (effect_value / 100)), 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif effect_type == EFFECT_CONTRAST_INC:
        img = np.clip((img.astype(np.float32) + ((127 + img.astype(np.float32)) * (effect_value / 100))), 0, 255).astype(np.uint8)
    elif effect_type == EFFECT_CONTRAST_DEC:
        img = np.clip((img.astype(np.float32) + ((127 - img.astype(np.float32)) * (effect_value / 100))), 0, 255).astype(np.uint8)
    elif effect_type == EFFECT_SATURATE:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip((hsv[:, :, 1].astype(np.float32) * ((100 + effect_value) / 100)), 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif effect_type == EFFECT_DESATURATE:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip((hsv[:, :, 1].astype(np.float32) * ((100 - effect_value) / 100)), 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif effect_type == EFFECT_ADAPTIVE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, effect_value, 2)
        # frame = cv2.Laplacian(frame, cv2.CV_8U)
        # frame = cv2.medianBlur(frame, 15)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif effect_type == EFFECT_NORMALIZE:
        img = cv2.normalize(img, img, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    elif effect_type == EFFECT_DEVIATION:
        img = img.astype(np.float32) / 255
        img -= img.mean()
        img /= img.std()
        img = cv2.normalize(img, img, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = (img * 255).astype(np.uint8)
    elif effect_type == EFFECT_DEVIATION2:
        mean = img.mean(axis=(0, 1))
        std = img.std(axis=(0, 1))
        img = img.astype(np.float32)
        img[..., 0] -= mean[0]
        img[..., 1] -= mean[1]
        img[..., 2] -= mean[2]
        img[..., 0] /= std[0]
        img[..., 1] /= std[1]
        img[..., 2] /= std[2]
        img = cv2.normalize(img, img, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = img.astype(np.uint8)
    elif effect_type == EFFECT_TEST1:
        img_B = img[:, :, 0]
        img_G = img[:, :, 1]
        img_R = img[:, :, 2]
        img_BG = cv2.addWeighted(img_B, 0.5, img_G, 0.5, 0)
        img_BR = cv2.addWeighted(img_B, 0.5, img_R, 0.5, 0)
        img_GR = cv2.addWeighted(img_G, 0.5, img_R, 0.5, 0)
        img_BBGR = cv2.addWeighted(img_B, 0.5, img_GR, 0.5, 0)
        img_BGGR = cv2.addWeighted(img_G, 0.5, img_BR, 0.5, 0)
        img_BGRR = cv2.addWeighted(img_R, 0.5, img_BG, 0.5, 0)
        # img = img_BBGR | img_BGGR | img_BGRR
        img = cv2.merge((img_BBGR, img_BGGR, img_BGRR))
    elif effect_type == EFFECT_TEST2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # img = img_gray | img_hsv[:, :, 2] | img_lab[:, :, 0]
        img = cv2.merge((img_gray, img_hsv[:, :, 2], img_lab[:, :, 0]))
    elif effect_type == EFFECT_FLIP:
        img = cv2.flip(img, 1)

    ''' if (effect_type == EFFECT_IN_SHAPE) and (effect_value[3] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)'''

    return img

def loadClassesTodatasets(dataset_names, extend_dataset_list, model_tune):  #  , add_channels, extend_dataset):
    global effect_print

    classes_source_path_list = [dataset_sources_path + '/' + dsn for dsn in dataset_names]

    # {"effects": [(EFFECT_NONE, "")], "model_tune": MODEL_TUNE_DEFAULT, "options": TASK_OPTIONS_DEFAULT},

    face_width = model_tune["in_shape"][1]
    face_height = model_tune["in_shape"][2]
    face_channels = model_tune["in_shape"][3]

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

    '''import zipfile
    from PIL import Image

    imgzip = zipfile.ZipFile("100-Test.zip")
    inflist = imgzip.infolist()

    for f in inflist:
        ifile = imgzip.open(f)
        img = Image.open(ifile)
        print(img)
        # display(img)'''

    '''import zipfile
    import cv2
    import numpy as np

    with zipfile.ZipFile('test.zip', 'r') as zfile:
        data = zfile.read('test.jpg')

    img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)'''

    #for class_path in class_pathes:
    for class_idx, class_name in enumerate(CLASS_NAMES):
        for classes_source_path in classes_source_path_list:
            #class_name = os.path.basename(os.path.normpath(class_path))
            #class_idx = CLASS_NAMES.index(class_name)
            class_path = classes_source_path + '/' + class_name
            files_list = os.listdir(class_path)
            print("Parsing class " + class_name + " of size " + str(len(files_list)) + " with effect " + effect_print + " from " + class_path)
            final_class_idx = class_idx
            new_class_idx = CLASSES_MAP[class_idx]
            # if new_class_idx != class_idx:
            new_class_name = CLASS_NAMES[new_class_idx]
            print("Mapping class {} to {}".format(class_name, new_class_name))
            # class_name = new_class_name
            # class_idx = new_class_idx
            final_class_idx = class_indexes.index(new_class_idx)
            for fidx, file in enumerate(files_list):
                if fidx > MAX_CLASS_IMAGES:
                    break
                if file.endswith(".jpg"):
                    file_name, file_ext = file.split('.')
                    #print("Parsing class " + class_name + " with file " + str(fidx) + " of " + str(len(files_list)) + " with effect " + effect_print + " from " + class_path)
                    #frame_timestamp, score, timestamp = file_name.split('_')
                    object_path = findParsedFile(file_name + '.txt')
                    #with open(objects_source_path + '/' + file_name + '.txt') as json_file:
                    # print("For", file_name, "found", object_path)
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
                            res_roi = cv2.resize(res_roi, (face_width, face_height), cv2.INTER_CUBIC)
                            res_roi_original = res_roi

                            for expand_dataset_list in extend_dataset_list:
                                res_roi = None
                                for effects_list in expand_dataset_list:
                                    add_to_res_roi = res_roi_original.copy()
                                    add_to_res_roi = applyEffectsList(add_to_res_roi, effects_list)
                                    if res_roi is None:
                                        res_roi = add_to_res_roi
                                    else:
                                        res_roi = np.concatenate((res_roi, add_to_res_roi), axis=2)

                                '''if ADD_CHANNELS:
                                    for add_channels_sequence in add_channels:
                                        add_to_res_roi = res_roi_original.copy()
                                        add_to_res_roi = applyEffectsList(add_to_res_roi, add_channels_sequence)
                                        res_roi = np.concatenate((res_roi, add_to_res_roi), axis=2)'''
                                #cv2.imshow("123", res_roi)
                                #cv2.imwrite(temp_path + '/' + file_name + '_face_roi.jpg', res_roi)
                                #res_roi = res_roi.reshape((1, 3, ) + res_roi.shape[:2])  # this is a Numpy array with shape (1, 3, 150, 150)
                                #res_roi = res_roi.reshape((3,) + res_roi.shape[:2])  # this is a Numpy array with shape (3, 150, 150)
                                #res_roi = res_roi.astype(np.float) # / 255
                                # res_roi = res_roi.copy()
                                lst_x_dataset.append(res_roi)
                                #lst_res_roi = datagen.flow(res_roi, batch_size=1)  # , save_to_dir='preview', save_prefix='cat', save_format='jpeg')
                                #lst_x_dataset.append(lst_res_roi)
                                y_set = np.zeros(num_classes)
                                y_set[final_class_idx] = 1
                                lst_y_dataset.append(y_set)
                                '''if EXTEND_DATASET:
                                    for extend_effect_sequence in extend_dataset:
                                        res_roi = res_roi_original.copy()
                                        res_roi = applyEffectsList(res_roi, extend_effect_sequence)
                                        lst_x_dataset.append(res_roi)
                                        y_set = np.zeros(num_classes)
                                        y_set[final_class_idx] = 1
                                        lst_y_dataset.append(y_set)'''
                        except:
                            print("Failed to process " + frame_path + " !!!")

    # return (lst_x_dataset, lst_y_dataset)

    #x_train = x_train.map(lambda x, y: (data_augmentation(x, training=True), y))
    x_dataset, y_dataset = np.array(lst_x_dataset), np.array(lst_y_dataset)
    lst_x_dataset, lst_y_dataset = [], []
    gc.collect()
    if USE_PRECISION == int(16):
        x_dataset, y_dataset = x_dataset.astype(np.float16), y_dataset.astype(np.float16)
    elif USE_PRECISION == int(64):
        x_dataset, y_dataset = x_dataset.astype(np.float64), y_dataset.astype(np.float64)
    gc.collect()
    return (x_dataset, y_dataset)
    #tmp = list(zip(lst_x_dataset, lst_y_dataset))
    #random.shuffle(tmp)
    #lst_x_dataset, lst_y_dataset = zip(*tmp)
    #return (np.array(lst_x_dataset), np.array(lst_y_dataset))

def make_model(input_shape, num_classes, add_dense=[], features_scale_factor=1, conv_pyramid=[128, 256, 512, 728, 1024]):
    from tensorflow.keras import layers
    from tensorflow import keras
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    #x = inputs
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(round(32 * features_scale_factor), 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(round(64 * features_scale_factor), 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # for size in [128, 256, 512, 728]:
    for size in conv_pyramid[:-1]:
        size = round(size * features_scale_factor)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # x = layers.SeparableConv2D(round(1024 * features_scale_factor), 3, padding="same")(x)
    x = layers.SeparableConv2D(round(conv_pyramid[-1] * features_scale_factor), 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    '''if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:'''
    activation = "softmax"
    units = num_classes

    x = layers.Dropout(0.5)(x)
    for dense_size in add_dense:
        x = layers.Dense(dense_size, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

def printTestResults(min_val_loss, max_val_accuracy, accuracy_simple):
    global test_results, super_max_val_accuracy, super_max_val_accuracy_key, super_min_val_loss, super_min_val_loss_key, super_max_combined, super_max_combined_key, super_accuracy_simple, super_accuracy_simple_key
    print("Test results at " + datetime.now().strftime("%Y.%m.%d %H:%M"))
    # test_results
    # print(test_results)
    print("{:<100} {:<17} {:<13} {:<13} {:<16}".format('Effect', 'max_val_accuracy', 'min_val_loss', 'max_combined', 'accuracy_simple'))
    for k, v in test_results.items():
        if v["max_val_accuracy"] > super_max_val_accuracy:
            super_max_val_accuracy = v["max_val_accuracy"]
            super_max_val_accuracy_key = k
        if v["min_val_loss"] < super_min_val_loss:
            super_min_val_loss = v["min_val_loss"]
            super_min_val_loss_key = k
        if v["max_combined"] > super_max_combined:
            super_max_combined = v["max_combined"]
            super_max_combined_key = k
        if v["accuracy_simple"] > super_accuracy_simple:
            super_accuracy_simple = v["accuracy_simple"]
            super_accuracy_simple_key = k
        '''max_combined = v["max_val_accuracy"] + 1 - v["min_val_loss"]
        if max_combined > super_max_combined:
            super_max_combined = max_combined
            super_max_combined_key = k'''
        max_val_accuracy = round(v["max_val_accuracy"], 3)
        min_val_loss = round(v["min_val_loss"], 3)
        max_combined = round(v["max_combined"], 3)
        accuracy_simple = round(v["accuracy_simple"], 3)
        print("{:<20} {:<17} {:<13} {:<13} {:<16}".format(k, max_val_accuracy, min_val_loss, max_combined, accuracy_simple))
    # input("Press Enter to continue...")
    # super_max_val_accuracy = round(super_max_val_accuracy, 3)
    # super_min_val_loss = round(super_min_val_loss, 3)
    # super_max_combined = round(super_max_combined, 3)
    # super_accuracy_simple = round(super_accuracy_simple, 3)

    print("super_max_val_accuracy={:<10} super_max_val_accuracy_key={} super_min_val_loss={:<10} super_min_val_loss_key={}".format(
            round(super_max_val_accuracy, 3), super_max_val_accuracy_key, round(super_min_val_loss, 3), super_min_val_loss_key))
    print("super_max_combined={:<10} super_max_combined_key={} super_max_combined_max_val_accuracy={:<10} super_max_combined_min_val_loss={:<10}".format(
            round(super_max_combined, 3), super_max_combined_key,
            round(test_results[super_max_combined_key]["max_val_accuracy"] if super_max_combined_key != '' else 0, 3),
            round(test_results[super_max_combined_key]["min_val_loss"] if super_max_combined_key != '' else 1, 3)))
    print("super_accuracy_simple={:<10} super_accuracy_simple_key={}".format(
            round(super_accuracy_simple, 3), super_accuracy_simple_key))

def formatEffectString(effect_type, effect_value):
    effect_print = effect_type
    if effect_value != "":
        effect_print += str(effect_value)
    return effect_print

# def runTest(effect_type, effect_value):
def taskTest(test_task, output_test_path):
    from tensorflow import keras
    from tensorflow.keras import backend as K

    import tensorflow as tf

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if USE_PRECISION == int(16):
        K.set_floatx('float16')
    elif USE_PRECISION == int(64):
        K.set_floatx('float64')

    global effect_print, test_results, x_out, y_out, last_effect_print_right, x_out_test_simple, y_out_test_simple

    # tf.config.experimental.set_memory_growth(physical_devices[0], False)
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # MODEL_TUNE_DEFAULT = {"scale_features": 1, "valid_proportion": 20, "add_dense": [], "in_shape": (1, 128, 128, 3), "conv_pyramid": [128, 256, 512, 728, 1024]}

    extend_dataset_list = test_task["effects"]

    sub_name = ""

    if "sub_name" in test_task:
        sub_name = test_task["sub_name"]

    model_name = MODEL_NAME + ("_" if MODEL_NAME != "" else "")  # + sub_name + ("_" if sub_name != "" else "")

    '''add_channels = []
    if "add_channels" in test_task:
        add_channels = test_task["add_channels"]'''

    '''extend_dataset = []
    if "extend_dataset" in test_task:
        extend_dataset = test_task["extend_dataset"]'''

    # super_test_images_count = SUPER_TEST_IMAGES_COUNT * (1 + len(extend_dataset))

    if "model_tune" in test_task:
        model_tune = test_task["model_tune"]
    else:
        model_tune = MODEL_TUNE_DEFAULT

    if "options" in test_task:
        options = test_task["options"]
    else:
        options = TASK_OPTIONS_DEFAULT

    if "dataset_names" in test_task:
        dataset_names = test_task["dataset_names"]
    else:
        dataset_names = DATASET_NAMES_DEFAULT

    face_width = model_tune["in_shape"][1]
    face_height = model_tune["in_shape"][2]
    face_channels = model_tune["in_shape"][3]

    # face_channels *= 1 + len(add_channels)
    face_channels *= len(extend_dataset_list[0])

    # add_dense = ""
    # for dense in model_tune["add_dense"]:
    #     add_dense += "-dense{}".format(dense)

    add_dense_str = "d".join(str(dense_size) for dense_size in model_tune["add_dense"])
    if add_dense_str != "":
        add_dense_str = "-" + add_dense_str

    conv_pyramid_str = "l".join(str(conv_layer_size) for conv_layer_size in model_tune["conv_pyramid"])
    if conv_pyramid_str != "":
        conv_pyramid_str = "-" + conv_pyramid_str

    effects_str = ""

    # if len(effects) > 0:
    #     effects_str += "pre-"
    for expand_dataset_list in extend_dataset_list:
        effects_str += "-"
        for effects_list in expand_dataset_list:
            effects_str += "-"
            for e_type, e_value in effects_list:
                effects_str += formatEffectString(e_type, e_value)

    if len(effects_str) > 150:
        effects_str = ""

    '''if len(add_channels) > 0:
        effects_str += "-ch-"
    else:
        effects_str += "-"
    for effs in add_channels:
        for e_type, e_value in effs:
            effects_str += formatEffectString(e_type, e_value)
        effects_str += "-"

    if len(extend_dataset) > 0:
        effects_str += "ex-"
    for effs in extend_dataset:
        for e_type, e_value in effs:
            effects_str += formatEffectString(e_type, e_value)
        effects_str += "-"'''

    effect_print_left = "{sub_name}{scale_features}-{valid_proportion}-{test_proportion}{add_dense}{conv_pyramid}".format(
        sub_name=(sub_name + ("-" if sub_name != "" else "")),
        scale_features=model_tune["scale_features"],
        valid_proportion=options["valid_proportion"],
        test_proportion=options["test_proportion"],
        add_dense=add_dense_str,
        conv_pyramid=conv_pyramid_str,
    )

    effect_print_right = "{face_width}x{face_height}x{face_channels}{effects_str}".format(
        face_width=face_width,
        face_height=face_height,
        face_channels=face_channels,
        effects_str=effects_str,
    )

    effect_print = effect_print_left + "-" + effect_print_right

    print("Training test model for " + effect_print)

    if (x_out is None) or (last_effect_print_right != effect_print_right):
        # x_out, y_out = loadClassesTodatasets(classes_source_path)
        x_train, y_train = None, None
        x_val, y_val = None, None
        x_test, y_test = None, None
        del x_test_simple
        del y_test_simple
        x_test_simple, y_test_simple = None, None
        lst_x_dataset, lst_y_dataset = None, None
        tmp = None
        del x_train
        del y_out
        x_out, y_out = None, None
        gc.collect()
        x_out, y_out = loadClassesTodatasets(dataset_names, extend_dataset_list, model_tune)  # , add_channels, extend_dataset)
        if USE_WORKTESTSET:
            x_test_simple, y_test_simple = loadClassesTodatasets(DATASET_NAMES_WORKTESTSET, extend_dataset_list, model_tune)  # add test dataset
        '''lst_x_dataset, lst_y_dataset = loadClassesTodatasets(dataset_names, extend_dataset_list, model_tune)  #  , add_channels, extend_dataset)
        lst_x_dataset, lst_y_dataset = loadClassesTodatasets(DATASET_NAMES_WORKTESTSET, extend_dataset_list, model_tune)  #  add test dataset
        x_dataset, y_dataset = np.array(lst_x_dataset), np.array(lst_y_dataset)
        if USE_PRECISION == int(16):
            x_out, y_out = x_dataset.astype(np.float16), y_dataset.astype(np.float16)
        elif USE_PRECISION == int(64):
            x_out, y_out = x_dataset.astype(np.float64), y_dataset.astype(np.float64)'''
        # x_test_simple, y_test_simple = x_out[-super_test_images_count:], y_out[-super_test_images_count:]
        # x_out, y_out = x_out[:-super_test_images_count], y_out[:-super_test_images_count]

    # last_effect_type, last_effect_value = effect_type, effect_value
    last_effect_print_right = effect_print_right

    proportion_val = options["valid_proportion"]
    count_train = int(len(x_out) * (100 - proportion_val - PORTION_TEST) / 100)
    count_val = int(len(x_out) * proportion_val / 100)
    count_test = int(len(x_out) * PORTION_TEST / 100)

    total_runs = options["runs"]
    if options["ext_epochs"] > 0:
        total_runs += 1

    model_best_simple_accuracy = None

    for run_idx in range(total_runs):

        print("||| Run", run_idx + 1, "for test model", effect_print)

        '''current_test_best_simple_accuracy = 0.1234
        current_test_best_simple_loss = 0.5678
        current_val_accuracy = 0.0123
        current_val_loss = 0.0456
        epochs = 123
        testpath = output_test_path + "/ta{0:03d}".format(int(round(current_test_best_simple_accuracy * 1000, 0))) + "tl{0:03d}".format(int(round(current_test_best_simple_loss * 1000, 0))) + "_i_va{:.3f}".format(round(current_val_accuracy, 3)) + "vl{:.3f}".format(round(current_val_loss, 3)) + "_r{0:02d}".format(run_idx + 1) + "e{0:03d}".format(epochs) + '_' + model_name + effect_print + '.h5'
        print(testpath)'''

        tmp = list(zip([x for x in x_out], [y for y in y_out]))
        x_out, y_out = None, None
        gc.collect()
        random.shuffle(tmp)
        lst_x_dataset, lst_y_dataset = zip(*tmp)
        del tmp
        tmp = None
        gc.collect()
        x_out, y_out = np.array(lst_x_dataset), np.array(lst_y_dataset)
        del lst_x_dataset
        del lst_y_dataset
        lst_x_dataset, lst_y_dataset = None, None
        gc.collect()

        x_train = x_out[:count_train]
        y_train = y_out[:count_train]
        x_val = x_out[count_train:count_train + count_val]
        y_val = y_out[count_train:count_train + count_val]
        x_test = x_out[-count_test:]
        y_test = y_out[-count_test:]

        if not USE_WORKTESTSET:
            x_test_simple, y_test_simple = x_test, y_test

        if (options["ext_epochs"] == 0) or (run_idx < total_runs - 1):
            model = make_model(
                input_shape=(face_width, face_height, face_channels),
                num_classes=num_classes,
                add_dense=model_tune["add_dense"],
                features_scale_factor=model_tune["scale_features"],
                conv_pyramid=model_tune["conv_pyramid"],
            )
            # keras.utils.plot_model(model, show_shapes=True)
            # opt = tf.train.AdamOptimizer(1e-3, epsilon=1e-4)
            model.compile(
                optimizer=(keras.optimizers.Adam(1e-3) if USE_PRECISION == int(32) else keras.optimizers.Adam(1e-3, epsilon=1e-4)),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            epochs = options["epochs"]
        else:
            if model_best_simple_accuracy is None:
                continue
            model = model_best_simple_accuracy
            model_best_simple_accuracy = None
            epochs = options["ext_epochs"]

        # callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5")]

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
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M")

        callbacks = []
        # a{accuracy:.2f}l{loss:.2f}
        # str(face_width) + 'x' + str(face_height) + 'x' + str(face_channels) + '_' +
        # "_" + time_stamp +
        if SAVE_MODELS:
            if SAVE_BEST_LOSS_CHECKPOINTS:
                callbacks.append(keras.callbacks.ModelCheckpoint(
                    filepath=output_test_path + "/i_va{val_accuracy:.5f}vl{val_loss:.5f}_r" + "{:02d}".format(run_idx + 1) + "e{epoch:04d}_" + model_name + effect_print + ".h5",
                    save_weights_only=False,
                    monitor='val_loss',
                    mode='auto',  #'max',
                    save_best_only=True))
            if SAVE_BEST_ACC_CHECKPOINTS:
                callbacks.append(keras.callbacks.ModelCheckpoint(
                    filepath=output_test_path + "/i_va{val_accuracy:.5f}vl{val_loss:.5f}_r" + "{:02d}".format(run_idx + 1) + "e{epoch:04d}_" + model_name + effect_print + ".h5",
                    save_weights_only=False,
                    monitor='val_accuracy',
                    mode='auto',
                    save_best_only=True))

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
            batch_size=BATCH_SIZE,
            epochs=epochs,
            #validation_data=(list(np.moveaxis(x_val, -1, 0)), y_val),
            validation_data=(x_val, y_val),
            #verbose=1,
            shuffle=True,
            callbacks=callbacks,
            #validation_split=0.2,
        )

        train_loss, val_loss, train_accuracy, val_accuracy = history.history['loss'], history.history['val_loss'], history.history['accuracy'], history.history['val_accuracy']

        max_val_accuracy = max(val_accuracy)
        min_val_loss = min(val_loss)

        max_combined = max([v_acc + 1 - v_loss for v_acc, v_loss in zip(val_accuracy, val_loss)])

        results_val = model.evaluate(x_val, y_val, batch_size=1)
        print("results after fit validation loss, validation acc:", results_val)

        current_val_accuracy = results_val[1]
        current_val_loss = results_val[0]

        results_super = model.evaluate(x_test_simple, y_test_simple, batch_size=1)
        print("results after fit super test simple loss, test simple acc:", results_super)

        current_test_best_simple_accuracy = results_super[1]
        current_test_best_simple_loss = results_super[0]

        if (effect_print not in test_results) or (current_test_best_simple_accuracy > test_results[effect_print]["accuracy_simple"]):
            model_best_simple_accuracy = model

        if effect_print in test_results:
            test_results[effect_print] = {"max_val_accuracy": max(test_results[effect_print]["max_val_accuracy"], max_val_accuracy), "min_val_loss": min(test_results[effect_print]["min_val_loss"], min_val_loss), "max_combined": max(test_results[effect_print]["max_combined"], max_combined), "accuracy_simple": max(test_results[effect_print]["accuracy_simple"], current_test_best_simple_accuracy)}
        else:
            test_results[effect_print] = {"max_val_accuracy": max_val_accuracy, "min_val_loss": min_val_loss, "max_combined": max_combined, "accuracy_simple": current_test_best_simple_accuracy}

        if SAVE_MODELS:
            # '_' + str(face_width) + 'x' + str(face_height) + 'x' + str(face_channels) + '_' + time_stamp +
            # model.save_weights('output/tests/result_mask_model_' + 'effect_' + effect_print + '_' + MODEL_NAME + '_' + str(face_width) + '_' + str(face_height) + '_' + str(face_channels) + '__' + time_stamp + '.h5')
            try:
                model.save(output_test_path + "/ta{0:03d}".format(int(round(current_test_best_simple_accuracy * 1000, 0))) + "tl{0:03d}".format(int(round(current_test_best_simple_loss * 1000, 0))) + "_i_va{:.5f}".format(round(current_val_accuracy, 3)) + "vl{:.5f}".format(round(current_val_loss, 3)) + "_r{0:02d}".format(run_idx + 1) + "e{0:04d}".format(epochs) + '_' + model_name + effect_print + '.h5')
            except:
                pass
            # output/tests/mask_model_" + MODEL_NAME + "_effect_" + effect_print + "_best_val_accuracy_{val_accuracy:.3f}_val_loss_{val_loss:.3f}_epoch_{epoch:03d}_loss_{loss:.3f}_accuracy_{accuracy:.3f}_" + str(face_width) + '_' + str(face_height) + '_' + str(face_channels) + '__' + time_stamp + ".h5"

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

        #results = model.evaluate(list(np.moveaxis(x_test, -1, 0)), y_test, batch_size=1)
        #results = model.evaluate(list(np.moveaxis(x_val, -1, 0)), y_val, batch_size=1)

        #results = model.evaluate(x_val, y_val, batch_size=1)
        #print("test loss, test acc:", results)

        #predictions = model.predict(list(np.moveaxis(x_test, -1, 0)))
        #predictions = model.predict(list(np.moveaxis(x_val, -1, 0)))
        #predictions = model.predict(x_train[-1].reshape((1, ) + x_train[-1].shape))
        predictions = model.predict(x_test_simple)
        print("predictions shape:", predictions.shape)
        # predictions
        #print("prediction [0, 0]:", predictions[0, 0], " should be:", y_test[0, 0, 0])
        print("prediction [0, 0]:", predictions[0], " should be:", y_test_simple[0])


        del model
        model = None
        del history
        history = None
        gc.collect()


        intermediate_models_filenames_list = glob.glob1(output_test_path, "i_*")
        for intermediate_filename in intermediate_models_filenames_list:
            model_intermediate = keras.models.load_model(os.path.join(output_test_path, intermediate_filename))
            results_super_intermediate = model_intermediate.evaluate(x_test_simple, y_test_simple, batch_size=1)
            print("results model_best_val_accuracy super test simple loss, test simple acc:", results_super_intermediate)
            # os.path.basename(
            if results_super_intermediate[1] > current_test_best_simple_accuracy:
                current_test_best_simple_accuracy = results_super_intermediate[1]
                model_best_simple_accuracy = model_intermediate
            rename_intermediate_filename = "ta{0:03d}".format(int(round(results_super_intermediate[1] * 1000, 0))) + "tl{0:03d}".format(int(round(results_super_intermediate[0] * 1000, 0))) + "_" + intermediate_filename
            try:
                os.rename(os.path.join(output_test_path, intermediate_filename), os.path.join(output_test_path, rename_intermediate_filename))
            except:
                os.remove(os.path.join(output_test_path, intermediate_filename))
            del model_intermediate
            model_intermediate = None
            gc.collect()

        if current_test_best_simple_accuracy > test_results[effect_print]["accuracy_simple"]:
            test_results[effect_print]["accuracy_simple"] = current_test_best_simple_accuracy

        '''best_val_accuracy_filepath_pattern = output_test_path + "/intermediate_mask_model_" + MODEL_NAME + "_effect_" + effect_print + "*_val_accuracy_{max_val_accuracy:.3f}*.h5".format(max_val_accuracy=max_val_accuracy)
        best_val_accuracy_simple_accuracy = 0
        best_val_accuracy_filepath_list = glob.glob(best_val_accuracy_filepath_pattern)
        if len(best_val_accuracy_filepath_list) > 0:
            best_val_accuracy_filepath = best_val_accuracy_filepath_list[0]

            model_best_val_accuracy = keras.models.load_model(best_val_accuracy_filepath)

            # results_super = model_best_val_accuracy.evaluate(list(np.moveaxis(x_test_simple, -1, 0)), y_test_simple, batch_size=1)
            results_super = model_best_val_accuracy.evaluate(x_test_simple, y_test_simple, batch_size=1)
            print("results model_best_val_accuracy super test simple loss, test simple acc:", results_super)

            best_val_accuracy_simple_accuracy = results_super[1]
            # predictions_simple = model_best_val_accuracy.predict(x_test_simple)
            # total_successful_predictions = 0
            # for idx, prediction_simple in enumerate(predictions_simple):
            #     if np.argmax(prediction_simple) == np.argmax(y_test_simple[idx]):
            #         total_successful_predictions += 1
            # accuracy_simple = total_successful_predictions / len(x_test_simple)
            # print("results super test simple accuracy:", round(accuracy_simple, 3))

            if best_val_accuracy_simple_accuracy > test_results[effect_print]["accuracy_simple"]:
                model_best_simple_accuracy = model_best_val_accuracy
                test_results[effect_print]["accuracy_simple"] = best_val_accuracy_simple_accuracy
                if SAVE_MODELS:
                    model_best_val_accuracy.save(output_test_path + "/best_test_mask_model_{model_name}_accuracy_simple_{accuracy_simple:.3f}_val_accuracy_{max_val_accuracy:.3f}_effect_{effect_print}_{face_width}x{face_height}x{face_channels}_{time_stamp}.h5".format(model_name=MODEL_NAME, effect_print=effect_print, max_val_accuracy=max_val_accuracy, accuracy_simple=best_val_accuracy_simple_accuracy, face_width=face_width, face_height=face_height, face_channels=face_channels, time_stamp=time_stamp))

            model_best_val_accuracy = None'''

        #input("Press Enter to continue...")

        #img = keras.preprocessing.image.load_img("PetImages/Cat/6779.jpg", target_size=image_size)
        #img_array = keras.preprocessing.image.img_to_array(img)
        #img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        #predictions = model.predict(img_array)
        #score = predictions[0]
        #print("This image is %.2f percent cat and %.2f percent dog." % (100 * (1 - score), 100 * score))

        printTestResults(min_val_loss, max_val_accuracy, current_test_best_simple_accuracy)

    del model_best_simple_accuracy
    model_best_simple_accuracy = None
    # del x_out
    # del y_out
    # del x_train
    # del y_train
    # del x_val
    # del y_val
    # del x_test
    # del y_test
    x_train, y_train = None, None
    x_val, y_val = None, None
    x_test, y_test = None, None
    gc.collect()


def taskRun():
    pass

if __name__ == '__main__':
    test_results = {}

    x_out = None
    y_out = None
    last_effect_print_right = None
    x_out_test_simple, y_out_test_simple = None, None

    super_max_val_accuracy = 0
    super_max_val_accuracy_key = ""
    super_min_val_loss = 1
    super_min_val_loss_key = ""

    super_max_combined = 0
    super_max_combined_key = ""

    super_accuracy_simple = 0
    super_accuracy_simple_key = ""

    effect_print = ""

    parsed_data_last_path_list = []

    freeze_support()
    # OUTPUT_TEST_PATH = pathToScriptFolder + "/output/tests/" + datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    OUTPUT_TEST_PATH = "G:/output/tests/" + datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    if not os.path.exists(OUTPUT_TEST_PATH):
        os.makedirs(OUTPUT_TEST_PATH)

    for test_task in TEST_TASKS_LIST:
        # for effect, value in test_task["effects"]:
        #     runTest(effect, value)
        taskTest(test_task, OUTPUT_TEST_PATH)
        '''p = Process(target=runTest, args=(test_task, OUTPUT_TEST_PATH))
        p.start()
        p.join()
        p.close()
        del p'''
        gc.collect()




















































'''DATASET_NAMES = [
    #'maskfacesnew',
    #'maskfacesnew3',
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
    #'vk_sources1',
]
EPOCHS = 10
TESTS_PER_EFFECT = 3
{'': {'max_val_accuracy': 0.8787625432014465, 'min_val_loss': 0.13014574348926544}, 'prop-20': {'max_val_accuracy': 0.8895859718322754, 'min_val_loss': 0.12816296517848969}, 'prop-40': {'max_val_accuracy': 0.8736280798912048, 'min_val_loss': 0.14555159211158752}, 'dense-1-1024': {'max_val_accuracy': 0.8804348111152649, 'min_val_loss': 0.14990253746509552}, 'dense-1-2048': {'max_val_accuracy': 0.8503344655036926, 'min_val_loss': 0.16665999591350555}, 'dense-1-4096': {'max_val_accuracy': 0.8591136932373047, 'min_val_loss': 0.1488523632287979}, 'dense-2-1024': {'max_val_accuracy': 0.8645485043525696, 'min_val_loss': 0.1524549126625061}, 'dense-3-1024': {'max_val_accuracy': 0.8386287689208984, 'min_val_loss': 0.1836743801832199}, 'shape-1-256-256': {'max_val_accuracy': 0.8695651888847351, 'min_val_loss': 0.1383947730064392}, 'shape-1-192-192': {'max_val_accuracy': 0.8628762364387512, 'min_val_loss': 0.15100687742233276}, 'shape-1-96-96': {'max_val_accuracy': 0.8708193898200989, 'min_val_loss': 0.14123471081256866}, 'shape-1-64-64': {'max_val_accuracy': 0.8570234179496765, 'min_val_loss': 0.15018194913864136}, 'shape-1-128-128': {'max_val_accuracy': 0.8591136932373047, 'min_val_loss': 0.1490272879600525}, 'gray': {'max_val_accuracy': 0.8494983315467834, 'min_val_loss': 0.15408854186534882}, 'hsv': {'max_val_accuracy': 0.8683110475540161, 'min_val_loss': 0.1493215709924698}, 'lab': {'max_val_accuracy': 0.8561872839927673, 'min_val_loss': 0.14625969529151917}, 'blur-9': {'max_val_accuracy': 0.8428093791007996, 'min_val_loss': 0.1755376011133194}, 'sharp-9': {'max_val_accuracy': 0.8758361339569092, 'min_val_loss': 0.13301987946033478}, 'bright-10': {'max_val_accuracy': 0.8800167441368103, 'min_val_loss': 0.1314009428024292}, 'dark-10': {'max_val_accuracy': 0.8791806101799011, 'min_val_loss': 0.13125963509082794}, 'cont-10': {'max_val_accuracy': 0.8687291145324707, 'min_val_loss': 0.14360156655311584}, 'decont-10': {'max_val_accuracy': 0.8846153616905212, 'min_val_loss': 0.1313290148973465}, 'sat-10': {'max_val_accuracy': 0.8800167441368103, 'min_val_loss': 0.13280636072158813}, 'desat-10': {'max_val_accuracy': 0.8754180669784546, 'min_val_loss': 0.12756942212581635}, 'ada-9': {'max_val_accuracy': 0.7286789417266846, 'min_val_loss': 0.30996695160865784}, 'norm': {'max_val_accuracy': 0.875, 'min_val_loss': 0.12706942856311798}, 'dev': {'max_val_accuracy': 0.8821070194244385, 'min_val_loss': 0.14376330375671387}, 'devv': {'max_val_accuracy': 0.8620401620864868, 'min_val_loss': 0.14879834651947021}}
Effect               max_val_accuracy min_val_loss
                     0.879      0.13      
prop-20              0.89       0.128     
prop-40              0.874      0.146     
dense-1-1024         0.88       0.15      
dense-1-2048         0.85       0.167     
dense-1-4096         0.859      0.149     
dense-2-1024         0.865      0.152     
dense-3-1024         0.839      0.184     
shape-1-256-256      0.87       0.138     
shape-1-192-192      0.863      0.151     
shape-1-96-96        0.871      0.141     
shape-1-64-64        0.857      0.15      
shape-1-128-128      0.859      0.149     
gray                 0.849      0.154     
hsv                  0.868      0.149     
lab                  0.856      0.146     
blur-9               0.843      0.176     
sharp-9              0.876      0.133     
bright-10            0.88       0.131     
dark-10              0.879      0.131     
cont-10              0.869      0.144     
decont-10            0.885      0.131     
sat-10               0.88       0.133     
desat-10             0.875      0.128     
ada-9                0.729      0.31      
norm                 0.875      0.127     
dev                  0.882      0.144     
devv                 0.862      0.149     
super_max_val_accuracy=0.89       super_max_val_accuracy_key=prop-20 super_min_val_loss=0.127      super_min_val_loss_key=norm
super_max_combined=1.761      super_max_combined_key=prop-20 super_max_combined_max_val_accuracy=0.89       super_max_combined_min_val_loss=0.128 '''







'''DATASET_NAMES = [
    #'maskfacesnew',
    #'maskfacesnew3',
    'maskfacesnewwork1',
    'maskfacesnewwork2',
    # 'maskfacesnewwork_toadd1',
    'false_faces_work1',
    'source_2020_11_25_1000ms',
    'source_2020_11_25_1000ms5max',
    'source_2020_11_25_200ms',
    'source_2020_11_25_3000ms',
    'source_2020_11_25_3000ms5max',
    'source_2020_11_25_300ms',
    'source_2020_11_25_noms',
    #'vk_sources1',
]
CLASSES_MAP = []
CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']
EPOCHS = 200
TESTS_PER_EFFECT = 3

Effect               max_val_accuracy min_val_loss
shape-1-128-128      0.926      0.112     
gray-3               0.929      0.102     
                     0.944      0.094     
super_max_val_accuracy=0.944      super_max_val_accuracy_key= super_min_val_loss=0.094      super_min_val_loss_key=
super_max_combined=1.85       super_max_combined_key= super_max_combined_max_val_accuracy=0.944      super_max_combined_min_val_loss=0.094 '''


'''DATASET_NAMES = [
    #'maskfacesnew',
    #'maskfacesnew3',
    'maskfacesnewwork1',
    'maskfacesnewwork2',
    # 'maskfacesnewwork_toadd1',
    'false_faces_work1',
    'source_2020_11_25_1000ms',
    'source_2020_11_25_1000ms5max',
    'source_2020_11_25_200ms',
    'source_2020_11_25_3000ms',
    'source_2020_11_25_3000ms5max',
    'source_2020_11_25_300ms',
    'source_2020_11_25_noms',
    #'vk_sources1',
]

CLASS_NAMES = ['mask', 'maskchin', 'masknone', 'masknose']
CLASSES_MAP = [0, 2, 2, 2]  # mask/masknone classes
EPOCHS = 200
TESTS_PER_EFFECT = 3

Effect               max_val_accuracy min_val_loss
shape-1-128-128      0.94       0.215     
gray-3               0.947      0.176     
                     0.944      0.177     
super_max_val_accuracy=0.947      super_max_val_accuracy_key=gray-3 super_min_val_loss=0.176      super_min_val_loss_key=gray-3
super_max_combined=1.755      super_max_combined_key=gray-3 super_max_combined_max_val_accuracy=0.947      super_max_combined_min_val_loss=0.176'''





'''DATASET_NAMES = [
    #'maskfacesnew',
    #'maskfacesnew3',
    'maskfacesnewwork1',
    'maskfacesnewwork2',
    # 'maskfacesnewwork_toadd1',
    'false_faces_work1',
    'source_2020_11_25_1000ms',
    'source_2020_11_25_1000ms5max',
    'source_2020_11_25_200ms',
    'source_2020_11_25_3000ms',
    'source_2020_11_25_3000ms5max',
    'source_2020_11_25_300ms',
    'source_2020_11_25_noms',
    #'vk_sources1',
    'worktestset_using_facemasknoses',
]

SUPER_TEST_IMAGES_COUNT = 480

CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']

EPOCHS = 200
TESTS_PER_EFFECT = 3

results super test loss, test acc: [0.6898380517959595, 0.706250011920929]
results super test simple accuracy: 0.706
Test results at 2021.01.19 01:16
{'': {'max_val_accuracy': 0.9424809217453003, 'min_val_loss': 0.08156746625900269, 'max_combined': 1.84774649143219, 'accuracy_simple': 0.50625}, 'scale-0.5': {'max_val_accuracy': 0.9459459185600281, 'min_val_loss': 0.07729559391736984, 'max_combined': 1.859638698399067, 'accuracy_simple': 0.5354166666666667}, 'scale-2': {'max_val_accuracy': 0.9452529549598694, 'min_val_loss': 0.09240670502185822, 'max_combined': 1.8382618427276611, 'accuracy_simple': 0.70625}}
Effect               max_val_accuracy  min_val_loss  max_combined  accuracy_simple 
                     0.942             0.082         1.848         0.506           
scale-0.5            0.946             0.077         1.86          0.535           
scale-2              0.945             0.092         1.838         0.706           
super_max_val_accuracy=0.946      super_max_val_accuracy_key=scale-0.5 super_min_val_loss=0.077      super_min_val_loss_key=scale-0.5
super_max_combined=1.86       super_max_combined_key=scale-0.5 super_max_combined_max_val_accuracy=0.946      super_max_combined_min_val_loss=0.077     
super_accuracy_simple=0.706      super_accuracy_simple_key=scale-2'''




'''DATASET_NAMES = [
    #'maskfacesnew',
    #'maskfacesnew3',
    'maskfacesnewwork1',
    'maskfacesnewwork2',
    # 'maskfacesnewwork_toadd1',
    'false_faces_work1',
    'source_2020_11_25_1000ms',
    'source_2020_11_25_1000ms5max',
    'source_2020_11_25_200ms',
    'source_2020_11_25_3000ms',
    'source_2020_11_25_3000ms5max',
    'source_2020_11_25_300ms',
    'source_2020_11_25_noms',
    #'vk_sources1',
    'worktestset_using_facemasknoses_mask_model_sort',  # 766 files
]

CLASSES_MAP = []
SUPER_TEST_IMAGES_COUNT = 766
CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']
MODEL_NAME = 'maskfacesnewwork12ffw1a1'
PORTION_VAL = 20
PORTION_TEST = 1
EPOCHS = 100
TESTS_PER_EFFECT = 3

results super test loss, test acc: [1.6765331029891968, 0.26631852984428406]
results super test simple accuracy: 0.266
Test results at 2021.01.24 17:00
{'': {'max_val_accuracy': 0.9501039385795593, 'min_val_loss': 0.07258547097444534, 'max_combined': 1.8708949014544487, 'accuracy_simple': 0.37989556135770236}, 'scale-0.5': {'max_val_accuracy': 0.9469854235649109, 'min_val_loss': 0.07037266343832016, 'max_combined': 1.8693362846970558, 'accuracy_simple': 0.3472584856396867}, 'scale-2': {'max_val_accuracy': 0.9594594836235046, 'min_val_loss': 0.06681206077337265, 'max_combined': 1.892647422850132, 'accuracy_simple': 0.44386422976501305}, 'prop-30': {'max_val_accuracy': 0.9487179517745972, 'min_val_loss': 0.07529278844594955, 'max_combined': 1.8661994487047195, 'accuracy_simple': 0.4725848563968668}, 'prop-40': {'max_val_accuracy': 0.942307710647583, 'min_val_loss': 0.08426055312156677, 'max_combined': 1.856487900018692, 'accuracy_simple': 0.2689295039164491}, 'dense-1-1024': {'max_val_accuracy': 0.9677754640579224, 'min_val_loss': 0.047652751207351685, 'max_combined': 1.9201227128505707, 'accuracy_simple': 0.3198433420365535}, 'dense-1-2048': {'max_val_accuracy': 0.9584199786186218, 'min_val_loss': 0.06596997380256653, 'max_combined': 1.881883256137371, 'accuracy_simple': 0.293733681462141}, 'dense-1-4096': {'max_val_accuracy': 0.9501039385795593, 'min_val_loss': 0.07023464888334274, 'max_combined': 1.8764656484127045, 'accuracy_simple': 0.44778067885117495}, 'dense-2-1024': {'max_val_accuracy': 0.9532224535942078, 'min_val_loss': 0.06382886320352554, 'max_combined': 1.8868780583143234, 'accuracy_simple': 0.38903394255874674}, 'dense-3-1024': {'max_val_accuracy': 0.9469854235649109, 'min_val_loss': 0.06568754464387894, 'max_combined': 1.875738948583603, 'accuracy_simple': 0.4425587467362924}, 'shape-1-128-128-1': {'max_val_accuracy': 0.9490644335746765, 'min_val_loss': 0.07247425615787506, 'max_combined': 1.873024582862854, 'accuracy_simple': 0.3289817232375979}, 'shape-1-256-256-3': {'max_val_accuracy': 0.9563409686088562, 'min_val_loss': 0.06069064140319824, 'max_combined': 1.895650327205658, 'accuracy_simple': 0.23237597911227154}, 'shape-1-192-192-3': {'max_val_accuracy': 0.9553014636039734, 'min_val_loss': 0.06216520816087723, 'max_combined': 1.890422374010086, 'accuracy_simple': 0.4164490861618799}, 'shape-1-96-96-3': {'max_val_accuracy': 0.9511434435844421, 'min_val_loss': 0.06293574720621109, 'max_combined': 1.8785905987024307, 'accuracy_simple': 0.46344647519582244}, 'shape-1-64-64-3': {'max_val_accuracy': 0.9501039385795593, 'min_val_loss': 0.07710443437099457, 'max_combined': 1.8705320283770561, 'accuracy_simple': 0.23759791122715404}, 'gray-3': {'max_val_accuracy': 0.9490644335746765, 'min_val_loss': 0.0791516900062561, 'max_combined': 1.8615967631340027, 'accuracy_simple': 0.556135770234987}, 'gray-0': {'max_val_accuracy': 0.9490644335746765, 'min_val_loss': 0.07013650238513947, 'max_combined': 1.8778884261846542, 'accuracy_simple': 0.5117493472584856}, 'hsv': {'max_val_accuracy': 0.952182948589325, 'min_val_loss': 0.08915267884731293, 'max_combined': 1.8603615388274193, 'accuracy_simple': 0.4151436031331593}, 'ycc': {'max_val_accuracy': 0.9542619585990906, 'min_val_loss': 0.06484797596931458, 'max_combined': 1.888199806213379, 'accuracy_simple': 0.5548302872062664}, 'lab-3': {'max_val_accuracy': 0.9428274631500244, 'min_val_loss': 0.0856487974524498, 'max_combined': 1.8507492318749428, 'accuracy_simple': 0.3263707571801567}, 'lab-0': {'max_val_accuracy': 0.94490647315979, 'min_val_loss': 0.08116792142391205, 'max_combined': 1.8511008322238922, 'accuracy_simple': 0.30156657963446476}, 'blur-3': {'max_val_accuracy': 0.952182948589325, 'min_val_loss': 0.06770750880241394, 'max_combined': 1.8830006048083305, 'accuracy_simple': 0.4086161879895561}, 'blur-1': {'max_val_accuracy': 0.957380473613739, 'min_val_loss': 0.06425870954990387, 'max_combined': 1.8818538561463356, 'accuracy_simple': 0.34595300261096606}, 'sharp-9': {'max_val_accuracy': 0.9532224535942078, 'min_val_loss': 0.06727389246225357, 'max_combined': 1.8859485611319542, 'accuracy_simple': 0.5652741514360313}, 'bright-10': {'max_val_accuracy': 0.9532224535942078, 'min_val_loss': 0.07724079489707947, 'max_combined': 1.8666261732578278, 'accuracy_simple': 0.47127937336814624}, 'dark-10': {'max_val_accuracy': 0.9490644335746765, 'min_val_loss': 0.07017268240451813, 'max_combined': 1.874908372759819, 'accuracy_simple': 0.3342036553524804}, 'cont-10': {'max_val_accuracy': 0.9511434435844421, 'min_val_loss': 0.06568719446659088, 'max_combined': 1.88129822909832, 'accuracy_simple': 0.21801566579634465}, 'decont-10': {'max_val_accuracy': 0.957380473613739, 'min_val_loss': 0.06442061066627502, 'max_combined': 1.8908808529376984, 'accuracy_simple': 0.587467362924282}, 'sat-10': {'max_val_accuracy': 0.9553014636039734, 'min_val_loss': 0.07575865834951401, 'max_combined': 1.8783986791968346, 'accuracy_simple': 0.48825065274151436}, 'desat-10': {'max_val_accuracy': 0.9584199786186218, 'min_val_loss': 0.061083823442459106, 'max_combined': 1.8943617567420006, 'accuracy_simple': 0.5365535248041775}, 'ada-3': {'max_val_accuracy': 0.8804574012756348, 'min_val_loss': 0.15480566024780273, 'max_combined': 1.714127391576767, 'accuracy_simple': 0.09530026109660575}, 'norm': {'max_val_accuracy': 0.9511434435844421, 'min_val_loss': 0.07222374528646469, 'max_combined': 1.8789196982979774, 'accuracy_simple': 0.29765013054830286}, 'dev': {'max_val_accuracy': 0.952182948589325, 'min_val_loss': 0.06829731911420822, 'max_combined': 1.8786881044507027, 'accuracy_simple': 0.35509138381201044}, 'devv': {'max_val_accuracy': 0.9511434435844421, 'min_val_loss': 0.0770372673869133, 'max_combined': 1.8626716807484627, 'accuracy_simple': 0.3563968668407311}, 'test1': {'max_val_accuracy': 0.9459459185600281, 'min_val_loss': 0.06502945721149445, 'max_combined': 1.87779800593853, 'accuracy_simple': 0.4908616187989556}, 'test2': {'max_val_accuracy': 0.9469854235649109, 'min_val_loss': 0.07959640026092529, 'max_combined': 1.852377861738205, 'accuracy_simple': 0.26631853785900783}}
Effect               max_val_accuracy  min_val_loss  max_combined  accuracy_simple 
                     0.95              0.073         1.871         0.38            
scale-0.5            0.947             0.07          1.869         0.347           
scale-2              0.959             0.067         1.893         0.444        +   
prop-30              0.949             0.075         1.866         0.473        +   
prop-40              0.942             0.084         1.856         0.269        -   
dense-1-1024         0.968             0.048         1.92          0.32            
dense-1-2048         0.958             0.066         1.882         0.294           
dense-1-4096         0.95              0.07          1.876         0.448        +   
dense-2-1024         0.953             0.064         1.887         0.389           
dense-3-1024         0.947             0.066         1.876         0.443        +   
shape-1-128-128-1    0.949             0.072         1.873         0.329           
shape-1-256-256-3    0.956             0.061         1.896         0.232        -   
shape-1-192-192-3    0.955             0.062         1.89          0.416           
shape-1-96-96-3      0.951             0.063         1.879         0.463        +   
shape-1-64-64-3      0.95              0.077         1.871         0.238        -   
gray-3               0.949             0.079         1.862         0.556        +++   
gray-0               0.949             0.07          1.878         0.512        ++   
hsv                  0.952             0.089         1.86          0.415           
ycc                  0.954             0.065         1.888         0.555        +++   
lab-3                0.943             0.086         1.851         0.326           
lab-0                0.945             0.081         1.851         0.302           
blur-3               0.952             0.068         1.883         0.409           
blur-1               0.957             0.064         1.882         0.346           
sharp-9              0.953             0.067         1.886         0.565        +++   
bright-10            0.953             0.077         1.867         0.471        +   
dark-10              0.949             0.07          1.875         0.334           
cont-10              0.951             0.066         1.881         0.218        --   
decont-10            0.957             0.064         1.891         0.587        ++++   
sat-10               0.955             0.076         1.878         0.488           
desat-10             0.958             0.061         1.894         0.537        +++   
ada-3                0.88              0.155         1.714         0.095        ----   
norm                 0.951             0.072         1.879         0.298           
dev                  0.952             0.068         1.879         0.355           
devv                 0.951             0.077         1.863         0.356           
test1                0.946             0.065         1.878         0.491        ++   
test2                0.947             0.08          1.852         0.266        -   
super_max_val_accuracy=0.968      super_max_val_accuracy_key=dense-1-1024 super_min_val_loss=0.048      super_min_val_loss_key=dense-1-1024
super_max_combined=1.92       super_max_combined_key=dense-1-1024 super_max_combined_max_val_accuracy=0.968      super_max_combined_min_val_loss=0.048     
super_accuracy_simple=0.641      super_accuracy_simple_key=lab-0

Process finished with exit code 0'''








'''DATASET_NAMES = [
    #'maskfacesnew',
    #'maskfacesnew3',
    'maskfacesnewwork1',
    'maskfacesnewwork2',
    # 'maskfacesnewwork_toadd1',
    'false_faces_work1',
    'source_2020_11_25_1000ms',
    'source_2020_11_25_1000ms5max',
    'source_2020_11_25_200ms',
    'source_2020_11_25_3000ms',
    'source_2020_11_25_3000ms5max',
    'source_2020_11_25_300ms',
    'source_2020_11_25_noms',
    #'vk_sources1',
    'worktestset_using_facemasknoses_mask_model_sort',  # 766 files
]

SUPER_TEST_IMAGES_COUNT = 766
CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']
FACE_WIDTH = 128
FACE_HEIGHT = 128
FACE_CHANNELS = 3
PORTION_VAL = 20
PORTION_TEST = 1
EPOCHS = 100
TESTS_PER_EFFECT = 3
results super test loss, test acc: [1.3955333232879639, 0.45430809259414673]
results super test simple accuracy: 0.454
Test results at 2021.01.25 09:07
{'': {'max_val_accuracy': 0.952182948589325, 'min_val_loss': 0.06407521665096283, 'max_combined': 1.8808312565088272, 'accuracy_simple': 0.4216710182767624}, 'sharp-1': {'max_val_accuracy': 0.9563409686088562, 'min_val_loss': 0.06224294379353523, 'max_combined': 1.8899400047957897, 'accuracy_simple': 0.4321148825065274}, 'sharp-3': {'max_val_accuracy': 0.9511434435844421, 'min_val_loss': 0.07240211218595505, 'max_combined': 1.873543806374073, 'accuracy_simple': 0.33159268929503916}, 'sharp-7': {'max_val_accuracy': 0.9594594836235046, 'min_val_loss': 0.061949167400598526, 'max_combined': 1.8975103162229061, 'accuracy_simple': 0.5274151436031331}, 'sharp-9': {'max_val_accuracy': 0.952182948589325, 'min_val_loss': 0.06858348101377487, 'max_combined': 1.8763229921460152, 'accuracy_simple': 0.46866840731070497}, 'sharp-11': {'max_val_accuracy': 0.9501039385795593, 'min_val_loss': 0.06781914085149765, 'max_combined': 1.8727800026535988, 'accuracy_simple': 0.24151436031331594}, 'sharp-13': {'max_val_accuracy': 0.9625779390335083, 'min_val_loss': 0.0625838190317154, 'max_combined': 1.897915169596672, 'accuracy_simple': 0.4621409921671018}, 'decont-6': {'max_val_accuracy': 0.9511434435844421, 'min_val_loss': 0.06729301065206528, 'max_combined': 1.879060685634613, 'accuracy_simple': 0.5939947780678851}, 'decont-8': {'max_val_accuracy': 0.9615384340286255, 'min_val_loss': 0.06618951261043549, 'max_combined': 1.888072445988655, 'accuracy_simple': 0.3002610966057441}, 'decont-10': {'max_val_accuracy': 0.952182948589325, 'min_val_loss': 0.06681095063686371, 'max_combined': 1.884902149438858, 'accuracy_simple': 0.37989556135770236}, 'decont-12': {'max_val_accuracy': 0.9480249285697937, 'min_val_loss': 0.08067107945680618, 'max_combined': 1.8517982587218285, 'accuracy_simple': 0.33550913838120106}, 'decont-14': {'max_val_accuracy': 0.957380473613739, 'min_val_loss': 0.054847270250320435, 'max_combined': 1.9025332033634186, 'accuracy_simple': 0.45300261096605743}, 'desat-6': {'max_val_accuracy': 0.9511434435844421, 'min_val_loss': 0.07279657572507858, 'max_combined': 1.8648333624005318, 'accuracy_simple': 0.4347258485639687}, 'desat-8': {'max_val_accuracy': 0.957380473613739, 'min_val_loss': 0.0625658705830574, 'max_combined': 1.8948146030306816, 'accuracy_simple': 0.45430809399477806}}
Effect               max_val_accuracy  min_val_loss  max_combined  accuracy_simple 
                     0.952             0.064         1.881         0.422           
sharp-1              0.956             0.062         1.89          0.432           
sharp-3              0.951             0.072         1.874         0.332           
sharp-7              0.959             0.062         1.898         0.527           
sharp-9              0.952             0.069         1.876         0.469           
sharp-11             0.95              0.068         1.873         0.242           
sharp-13             0.963             0.063         1.898         0.462           
decont-6             0.951             0.067         1.879         0.594           
decont-8             0.962             0.066         1.888         0.3             
decont-10            0.952             0.067         1.885         0.38            
decont-12            0.948             0.081         1.852         0.336           
decont-14            0.957             0.055         1.903         0.453           
desat-6              0.951             0.073         1.865         0.435           
desat-8              0.957             0.063         1.895         0.454           
super_max_val_accuracy=0.963      super_max_val_accuracy_key=sharp-13 super_min_val_loss=0.055      super_min_val_loss_key=decont-14
super_max_combined=1.903      super_max_combined_key=decont-14 super_max_combined_max_val_accuracy=0.957      super_max_combined_min_val_loss=0.055     
super_accuracy_simple=0.602      super_accuracy_simple_key=decont-8'''









'''DATASET_NAMES = [
    # 'maskfacesnew',  # not work
    # 'maskfacesnew3',  # not work
    'maskfacesnewwork1',
    'maskfacesnewwork2',
    # 'maskfacesnewwork_toadd1',  # not work
    'false_faces_work1',
    'source_2020_11_25_1000ms',
    'source_2020_11_25_1000ms5max',
    'source_2020_11_25_200ms',
    'source_2020_11_25_3000ms',
    'source_2020_11_25_3000ms5max',
    'source_2020_11_25_300ms',
    'source_2020_11_25_noms',
    # 'vk_sources1',  # not work
    'worktestset_using_facemasknoses_mask_model_sort',  # 766 files
]

CLASSES_MAP = []

CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']
FACE_WIDTH = 128
FACE_HEIGHT = 128
FACE_CHANNELS = 3
PORTION_VAL = 20
PORTION_TEST = 1
EPOCHS = 10
TESTS_PER_EFFECT = 5
EXT_EPOCHS = 100

MAX_CLASS_IMAGES = 999999
SAVE_MODELS = True
ADD_CHANNELS = False
EXTEND_DATASET = False
Test results at 2021.02.06 20:53
Effect                                                                                               max_val_accuracy  min_val_loss  max_combined  accuracy_simple 
1-20--128l256l512l728l1024-128x128x3-pre-gray3-add-ext- 0.938             0.09          1.845         0.778           
0.5-20--128l256l512l728l1024-128x128x3-pre-gray3-add-ext- 0.936             0.084         1.842         0.589           
2-20--128l256l512l728l1024-128x128x3-pre-gray3-add-ext- 0.949             0.074         1.872         0.471           
1-20--128l256l512l728l1024l2048-128x128x3-pre-gray3-add-ext- 0.942             0.096         1.834         0.508           
1-20--64l128l256l512l728l1024-128x128x3-pre-gray3-add-ext- 0.939             0.103         1.822         0.376           
1-20--256l512l728l1024-128x128x3-pre-gray3-add-ext- 0.937             0.1           1.835         0.697           
1-20--128l256l512l728-128x128x3-pre-gray3-add-ext- 0.946             0.077         1.864         0.547           
1-20-512-128l256l512l728l1024-128x128x3-pre-gray3-add-ext- 0.957             0.06          1.898         0.505           
1-20-1024-128l256l512l728l1024-128x128x3-pre-gray3-add-ext- 0.946             0.075         1.863         0.363           
1-20-2048-128l256l512l728l1024-128x128x3-pre-gray3-add-ext- 0.942             0.085         1.85          0.512           
1-20-512d512-128l256l512l728l1024-128x128x3-pre-gray3-add-ext- 0.949             0.068         1.873         0.611           
1-20-1024d1024-128l256l512l728l1024-128x128x3-pre-gray3-add-ext- 0.923             0.104         1.81          0.253           
super_max_val_accuracy=0.957      super_max_val_accuracy_key=1-20-512-128l256l512l728l1024-128x128x3-pre-gray3-add-ext- super_min_val_loss=0.06       super_min_val_loss_key=1-20-512-128l256l512l728l1024-128x128x3-pre-gray3-add-ext-
super_max_combined=1.898      super_max_combined_key=1-20-512-128l256l512l728l1024-128x128x3-pre-gray3-add-ext- super_max_combined_max_val_accuracy=0.957      super_max_combined_min_val_loss=0.06      
super_accuracy_simple=0.778      super_accuracy_simple_key=1-20--128l256l512l728l1024-128x128x3-pre-gray3-add-ext-

Test results at 2021.02.06 22:03
Effect                                                                                               max_val_accuracy  min_val_loss  max_combined  accuracy_simple 
1-20--128l256l512l728l1024-128x128x3-pre-gray3-add-ext- 0.94              0.083         1.855         0.337           
1-20--256l512l728l1024-128x128x3-pre-gray3-add-ext- 0.95              0.067         1.882         0.487           
super_max_val_accuracy=0.95       super_max_val_accuracy_key=1-20--256l512l728l1024-128x128x3-pre-gray3-add-ext- super_min_val_loss=0.067      super_min_val_loss_key=1-20--256l512l728l1024-128x128x3-pre-gray3-add-ext-
super_max_combined=1.882      super_max_combined_key=1-20--256l512l728l1024-128x128x3-pre-gray3-add-ext- super_max_combined_max_val_accuracy=0.95       super_max_combined_min_val_loss=0.067     
super_accuracy_simple=0.487      super_accuracy_simple_key=1-20--256l512l728l1024-128x128x3-pre-gray3-add-ext-'''





'''DATASET_NAMES = [
    # 'maskfacesnew',  # not work
    # 'maskfacesnew3',  # not work
    'maskfacesnewwork1',
    'maskfacesnewwork2',
    # 'maskfacesnewwork_toadd1',  # not work
    'false_faces_work1',
    'source_2020_11_25_1000ms',
    'source_2020_11_25_1000ms5max',
    'source_2020_11_25_200ms',
    'source_2020_11_25_3000ms',
    'source_2020_11_25_3000ms5max',
    'source_2020_11_25_300ms',
    'source_2020_11_25_noms',
    # 'vk_sources1',  # not work
    'worktestset_using_facemasknoses_mask_model_sort',  # 766 files
]

CLASSES_MAP = []

CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']

FACE_WIDTH = 128
FACE_HEIGHT = 128
FACE_CHANNELS = 3
PORTION_VAL = 20
PORTION_TEST = 1
EPOCHS = 20
TESTS_PER_EFFECT = 10
EXT_EPOCHS = 100
SAVE_MODELS = True
ADD_CHANNELS = True
EXTEND_DATASET = False

results model_best_val_accuracy super test simple loss, test simple acc: [1.527141809463501, 0.4712793827056885]
Test results at 2021.02.07 05:47
Effect                                                                                               max_val_accuracy  min_val_loss  max_combined  accuracy_simple 
1-20--128l256l512l728l1024-128x128x57-pre--add-gray3-hsv-ycc-lab3-test1-test2-ada3-blur9-sharp9-bright10-dark10-cont10-decont10-sat10-desat10-norm-dev-devv-ext- 0.947             0.063         1.884         0.615           
1-20--128l256l512l728l1024-128x128x6-pre-gray3-add-gray3decont10-ext- 0.949             0.074         1.864         0.556           
1-20--128l256l512l728l1024-128x128x6-pre-gray3-add-gray3sharp9-ext- 0.948             0.06          1.888         0.743           
1-20--128l256l512l728l1024-128x128x6-pre-gray3-add-gray3blur9-ext- 0.954             0.062         1.892         0.717           
1-20--128l256l512l728l1024-128x128x6-pre-gray3-add-gray3bright10-ext- 0.945             0.078         1.859         0.561           
1-20--128l256l512l728l1024-128x128x6-pre--add-gray3-ext- 0.957             0.062         1.895         0.679           
1-20--128l256l512l728l1024-128x128x3-pre--add-ext- 0.955             0.059         1.892         0.482           
super_max_val_accuracy=0.957      super_max_val_accuracy_key=1-20--128l256l512l728l1024-128x128x6-pre--add-gray3-ext- super_min_val_loss=0.059      super_min_val_loss_key=1-20--128l256l512l728l1024-128x128x3-pre--add-ext-
super_max_combined=1.895      super_max_combined_key=1-20--128l256l512l728l1024-128x128x6-pre--add-gray3-ext- super_max_combined_max_val_accuracy=0.957      super_max_combined_min_val_loss=0.062     
super_accuracy_simple=0.743      super_accuracy_simple_key=1-20--128l256l512l728l1024-128x128x6-pre-gray3-add-gray3sharp9-ext-'''






'''DATASET_NAMES = [
    # 'maskfacesnew',  # not work
    # 'maskfacesnew3',  # not work
    'maskfacesnewwork1',
    'maskfacesnewwork2',
    # 'maskfacesnewwork_toadd1',  # not work
    'false_faces_work1',
    'source_2020_11_25_1000ms',
    'source_2020_11_25_1000ms5max',
    'source_2020_11_25_200ms',
    'source_2020_11_25_3000ms',
    'source_2020_11_25_3000ms5max',
    'source_2020_11_25_300ms',
    'source_2020_11_25_noms',
    # 'vk_sources1',  # not work
    'worktestset_using_facemasknoses_mask_model_sort',  # 766 files
]

CLASSES_MAP = []

CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']


FACE_WIDTH = 128
FACE_HEIGHT = 128
FACE_CHANNELS = 3
PORTION_VAL = 20
PORTION_TEST = 1
EPOCHS = 20
TESTS_PER_EFFECT = 10
EXT_EPOCHS = 100
MAX_CLASS_IMAGES = 999999
SAVE_MODELS = True
ADD_CHANNELS = False


results model_best_val_accuracy super test simple loss, test simple acc: [0.8211026787757874, 0.4020887613296509]
Test results at 2021.02.07 15:37
Effect                                                                                               max_val_accuracy  min_val_loss  max_combined  accuracy_simple 
1-20--128l256l512l728l1024-128x128x3-pre--add-ext-gray3-blur3-sharp9-bright10-dark10-cont10-decont10-norm-dev-devv- 0.999             0.001         1.998         0.663           
1-20--128l256l512l728l1024-128x128x3-pre--add-ext- 0.949             0.075         1.859         0.676           
super_max_val_accuracy=0.999      super_max_val_accuracy_key=1-20--128l256l512l728l1024-128x128x3-pre--add-ext-gray3-blur3-sharp9-bright10-dark10-cont10-decont10-norm-dev-devv- super_min_val_loss=0.001      super_min_val_loss_key=1-20--128l256l512l728l1024-128x128x3-pre--add-ext-gray3-blur3-sharp9-bright10-dark10-cont10-decont10-norm-dev-devv-
super_max_combined=1.998      super_max_combined_key=1-20--128l256l512l728l1024-128x128x3-pre--add-ext-gray3-blur3-sharp9-bright10-dark10-cont10-decont10-norm-dev-devv- super_max_combined_max_val_accuracy=0.999      super_max_combined_min_val_loss=0.001     
super_accuracy_simple=0.676      super_accuracy_simple_key=1-20--128l256l512l728l1024-128x128x3-pre--add-ext-
'''





'''DATASET_NAMES = [
    # 'maskfacesnew',  # not work
    # 'maskfacesnew3',  # not work
    'maskfacesnewwork1',
    'maskfacesnewwork2',
    # 'maskfacesnewwork_toadd1',  # not work
    'false_faces_work1',
    'source_2020_11_25_1000ms',
    'source_2020_11_25_1000ms5max',
    'source_2020_11_25_200ms',
    'source_2020_11_25_3000ms',
    'source_2020_11_25_3000ms5max',
    'source_2020_11_25_300ms',
    'source_2020_11_25_noms',
    # 'vk_sources1',  # not work
    'worktestset_using_facemasknoses_mask_model_sort',  # 766 files
]

CLASSES_MAP = []

CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']

FACE_WIDTH = 128
FACE_HEIGHT = 128
FACE_CHANNELS = 3
PORTION_VAL = 20
PORTION_TEST = 1
EPOCHS = 20
TESTS_PER_EFFECT = 10
EXT_EPOCHS = 100
SAVE_MODELS = True
ADD_CHANNELS = False
EXTEND_DATASET = False


results model_best_val_accuracy super test simple loss, test simple acc: [1.7390007972717285, 0.2806788384914398]
Test results at 2021.02.08 09:25
Effect                                                                                               max_val_accuracy  min_val_loss  max_combined  accuracy_simple 
1-20--128l256l512l728l1024-128x128x3-pre--add-ext- 0.964             0.047         1.913         0.633           
1-20--128l256l512l728l1024-128x128x3-pre-gray3-add-ext- 0.942             0.079         1.861         0.685           
1-20--128l256l512l728l1024-128x128x3-pre-hsv-add-ext- 0.952             0.059         1.885         0.762           
1-20--128l256l512l728l1024-128x128x3-pre-ycc-add-ext- 0.945             0.076         1.866         0.654           
1-20--128l256l512l728l1024-128x128x3-pre-lab3-add-ext- 0.952             0.064         1.888         0.514           
1-20--128l256l512l728l1024-128x128x3-pre-blur5-add-ext- 0.959             0.055         1.904         0.559           
1-20--128l256l512l728l1024-128x128x3-pre-sharp9-add-ext- 0.967             0.04          1.927         0.594           
1-20--128l256l512l728l1024-128x128x3-pre-bright10-add-ext- 0.956             0.065         1.889         0.578           
1-20--128l256l512l728l1024-128x128x3-pre-dark10-add-ext- 0.955             0.05          1.901         0.641           
1-20--128l256l512l728l1024-128x128x3-pre-cont10-add-ext- 0.946             0.067         1.872         0.601           
1-20--128l256l512l728l1024-128x128x3-pre-decont10-add-ext- 0.95              0.065         1.879         0.486           
1-20--128l256l512l728l1024-128x128x3-pre-sat10-add-ext- 0.948             0.072         1.865         0.872           
1-20--128l256l512l728l1024-128x128x3-pre-desat10-add-ext- 0.956             0.058         1.899         0.783           
1-20--128l256l512l728l1024-128x128x3-pre-ada3-add-ext- 0.89              0.13          1.756         0.178           
1-20--128l256l512l728l1024-128x128x3-pre-norm-add-ext- 0.955             0.058         1.897         0.748           
1-20--128l256l512l728l1024-128x128x3-pre-dev-add-ext- 0.951             0.067         1.867         0.55            
1-20--128l256l512l728l1024-128x128x3-pre-devv-add-ext- 0.96              0.047         1.908         0.637           
1-20--128l256l512l728l1024-128x128x3-pre-test1-add-ext- 0.952             0.069         1.879         0.625           
1-20--128l256l512l728l1024-128x128x3-pre-test2-add-ext- 0.95              0.07          1.874         0.701           
super_max_val_accuracy=0.967      super_max_val_accuracy_key=1-20--128l256l512l728l1024-128x128x3-pre-sharp9-add-ext- super_min_val_loss=0.04       super_min_val_loss_key=1-20--128l256l512l728l1024-128x128x3-pre-sharp9-add-ext-
super_max_combined=1.927      super_max_combined_key=1-20--128l256l512l728l1024-128x128x3-pre-sharp9-add-ext- super_max_combined_max_val_accuracy=0.967      super_max_combined_min_val_loss=0.04      
super_accuracy_simple=0.872      super_accuracy_simple_key=1-20--128l256l512l728l1024-128x128x3-pre-sat10-add-ext-'''










'''DATASET_NAMES = [
    # 'maskfacesnew',  # not work
    # 'maskfacesnew3',  # not work
    'maskfacesnewwork1',
    'maskfacesnewwork2',
    # 'maskfacesnewwork_toadd1',  # not work
    'false_faces_work1',
    'source_2020_11_25_1000ms',
    'source_2020_11_25_1000ms5max',
    'source_2020_11_25_200ms',
    'source_2020_11_25_3000ms',
    'source_2020_11_25_3000ms5max',
    'source_2020_11_25_300ms',
    'source_2020_11_25_noms',
    # 'vk_sources1',  # not work
    'worktestset_using_facemasknoses_mask_model_sort',  # 766 files
]

CLASSES_MAP = []

CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']

FACE_WIDTH = 128
FACE_HEIGHT = 128
FACE_CHANNELS = 3
PORTION_VAL = 20
PORTION_TEST = 1
EPOCHS = 20
TESTS_PER_EFFECT = 10
EXT_EPOCHS = 100
MAX_CLASS_IMAGES = 999999
SAVE_MODELS = True
ADD_CHANNELS = False
EXTEND_DATASET = False


results model_best_val_accuracy super test simple loss, test simple acc: [1.7975420951843262, 0.19973890483379364]
Test results at 2021.02.08 18:54
Effect                                                                                               max_val_accuracy  min_val_loss  max_combined  accuracy_simple 
1-20--128l256l512l728l1024-128x128x3-pre-gray3-add-ext- 0.948             0.064         1.884         0.514           
1-20--128l256l512l728l1024-128x128x3-pre-gray3blur5-add-ext- 0.943             0.076         1.856         0.64            
1-20--128l256l512l728l1024-128x128x3-pre-gray3sharp9-add-ext- 0.948             0.078         1.857         0.578           
1-20--128l256l512l728l1024-128x128x3-pre-gray3bright10-add-ext- 0.948             0.068         1.877         0.688           
1-20--128l256l512l728l1024-128x128x3-pre-gray3dark10-add-ext- 0.939             0.081         1.851         0.781           
1-20--128l256l512l728l1024-128x128x3-pre-gray3cont10-add-ext- 0.947             0.059         1.887         0.757           
1-20--128l256l512l728l1024-128x128x3-pre-gray3decont10-add-ext- 0.936             0.091         1.824         0.612           
1-20--128l256l512l728l1024-128x128x3-pre-gray3sat10-add-ext- 0.953             0.062         1.885         0.735           
1-20--128l256l512l728l1024-128x128x3-pre-gray3desat10-add-ext- 0.936             0.093         1.829         0.578           
1-20--128l256l512l728l1024-128x128x3-pre-gray3norm-add-ext- 0.942             0.086         1.851         0.531           
1-20--128l256l512l728l1024-128x128x3-pre-gray3dev-add-ext- 0.937             0.083         1.848         0.772           
1-20--128l256l512l728l1024-128x128x3-pre-gray3devv-add-ext- 0.954             0.072         1.874         0.833           
super_max_val_accuracy=0.954      super_max_val_accuracy_key=1-20--128l256l512l728l1024-128x128x3-pre-gray3devv-add-ext- super_min_val_loss=0.059      super_min_val_loss_key=1-20--128l256l512l728l1024-128x128x3-pre-gray3cont10-add-ext-
super_max_combined=1.887      super_max_combined_key=1-20--128l256l512l728l1024-128x128x3-pre-gray3cont10-add-ext- super_max_combined_max_val_accuracy=0.947      super_max_combined_min_val_loss=0.059     
super_accuracy_simple=0.833      super_accuracy_simple_key=1-20--128l256l512l728l1024-128x128x3-pre-gray3devv-add-ext-'''









'''DATASET_NAMES = [
    # 'maskfacesnew',  # not work
    # 'maskfacesnew3',  # not work
    'maskfacesnewwork1',
    'maskfacesnewwork2',
    # 'maskfacesnewwork_toadd1',  # not work
    'false_faces_work1',
    'source_2020_11_25_1000ms',
    'source_2020_11_25_1000ms5max',
    'source_2020_11_25_200ms',
    'source_2020_11_25_3000ms',
    'source_2020_11_25_3000ms5max',
    'source_2020_11_25_300ms',
    'source_2020_11_25_noms',
    # 'vk_sources1',  # not work
    'worktestset_using_facemasknoses_mask_model_sort',  # 766 files
]

# SUPER_TEST_IMAGES_COUNT = 766

CLASSES_MAP = []

CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']


FACE_WIDTH = 128
FACE_HEIGHT = 128

FACE_CHANNELS = 3
PORTION_VAL = 20
PORTION_TEST = 20
EPOCHS = 20
TESTS_PER_EFFECT = 10
EXT_EPOCHS = 100
MAX_CLASS_IMAGES = 999999
SAVE_MODELS = True
SAVE_BEST_LOSS_CHECKPOINTS = True
SAVE_BEST_ACC_CHECKPOINTS = True
ADD_CHANNELS = False
EXTEND_DATASET = False


results model_best_val_accuracy super test simple loss, test simple acc: [0.19221699237823486, 0.9040358662605286]
Test results at 2021.02.09 00:49
Effect                                                                                               max_val_accuracy  min_val_loss  max_combined  accuracy_simple 
1-20--128l256l512l728l1024-128x128x3-pre--add-ext- 0.926             0.103         1.806         0.918           
1-20--128l256l512l728l1024-128x128x3-pre-gray3-add-ext- 0.922             0.104         1.818         0.928           
1-20--128l256l512l728l1024-128x128x3-pre--add-ext-flip- 0.927             0.09          1.834         0.921           
1-20--128l256l512l728l1024-128x128x3-pre-gray3-add-ext-flip- 0.92              0.104         1.803         0.911           
super_max_val_accuracy=0.927      super_max_val_accuracy_key=1-20--128l256l512l728l1024-128x128x3-pre--add-ext-flip- super_min_val_loss=0.09       super_min_val_loss_key=1-20--128l256l512l728l1024-128x128x3-pre--add-ext-flip-
super_max_combined=1.834      super_max_combined_key=1-20--128l256l512l728l1024-128x128x3-pre--add-ext-flip- super_max_combined_max_val_accuracy=0.927      super_max_combined_min_val_loss=0.09      
super_accuracy_simple=0.928      super_accuracy_simple_key=1-20--128l256l512l728l1024-128x128x3-pre-gray3-add-ext-'''




'''2021-02-10 11:37:32.510713: I tensorflow/core/common_runtime/bfc_allocator.cc:990] InUse at dbfabf700 of size 2048 next 34939
2021-02-10 11:37:32.510856: I tensorflow/core/common_runtime/bfc_allocator.cc:990] InUse at dbfabff00 of size 2048 next 34948
2021-02-10 11:37:32.511000: I tensorflow/core/common_runtime/bfc_allocator.cc:990] InUse at dbfac0700 of size 2048 next 34950
2021-02-10 11:37:32.511143: I tensorflow/core/common_runtime/bfc_allocator.cc:990] InUse at dbfac0f00 of size 2048 next 34957



2021-02-10 11:37:33.763359: I tensorflow/core/common_runtime/bfc_allocator.cc:998] 5 Chunks of size 33554432 totalling 160.00MiB
2021-02-10 11:37:33.763504: I tensorflow/core/common_runtime/bfc_allocator.cc:998] 1 Chunks of size 46513664 totalling 44.36MiB
2021-02-10 11:37:33.763649: I tensorflow/core/common_runtime/bfc_allocator.cc:998] 5 Chunks of size 67108864 totalling 320.00MiB
2021-02-10 11:37:33.763806: I tensorflow/core/common_runtime/bfc_allocator.cc:1002] Sum Total of in-use chunks: 8.47GiB
2021-02-10 11:37:33.763968: I tensorflow/core/common_runtime/bfc_allocator.cc:1004] total_region_allocated_bytes_: 9106106880 memory_limit_: 9106107123 available bytes: 243 curr_region_allocation_bytes_: 17179869184
2021-02-10 11:37:33.764224: I tensorflow/core/common_runtime/bfc_allocator.cc:1010] Stats: 
Limit:                  9106107123
InUse:                  9096676608
MaxInUse:               9096676608
NumAllocs:               348700117
MaxAllocSize:           1812824064

2021-02-10 11:37:33.766670: W tensorflow/core/common_runtime/bfc_allocator.cc:439] ****************************************************************************************************
2021-02-10 11:37:33.767106: W tensorflow/core/framework/op_kernel.cc:1753] OP_REQUIRES failed at depthwise_conv_op.cc:376 : Resource exhausted: OOM when allocating tensor with shape[32,256,32,32] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
Traceback (most recent call last):
  File "C:\Program Files\Python37\lib\contextlib.py", line 130, in __exit__
    self.gen.throw(type, value, traceback)
  File "C:\Program Files\Python37\lib\site-packages\tensorflow\python\ops\variable_scope.py", line 2805, in variable_creator_scope
    yield
  File "C:\Program Files\Python37\lib\site-packages\tensorflow\python\keras\engine\training.py", line 848, in fit
    tmp_logs = train_function(iterator)
  File "C:\Program Files\Python37\lib\site-packages\tensorflow\python\eager\def_function.py", line 580, in __call__
    result = self._call(*args, **kwds)
  File "C:\Program Files\Python37\lib\site-packages\tensorflow\python\eager\def_function.py", line 644, in _call
    return self._stateless_fn(*args, **kwds)
  File "C:\Program Files\Python37\lib\site-packages\tensorflow\python\eager\function.py", line 2420, in __call__
    return graph_function._filtered_call(args, kwargs)  # pylint: disable=protected-access
  File "C:\Program Files\Python37\lib\site-packages\tensorflow\python\eager\function.py", line 1665, in _filtered_call
    self.captured_inputs)
  File "C:\Program Files\Python37\lib\site-packages\tensorflow\python\eager\function.py", line 1746, in _call_flat
    ctx, args, cancellation_manager=cancellation_manager))
  File "C:\Program Files\Python37\lib\site-packages\tensorflow\python\eager\function.py", line 598, in call
    ctx=ctx)
  File "C:\Program Files\Python37\lib\site-packages\tensorflow\python\eager\execute.py", line 60, in quick_execute
    inputs, attrs, num_outputs)
tensorflow.python.framework.errors_impl.ResourceExhaustedError:  OOM when allocating tensor with shape[32,256,32,32] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
	 [[node model_29/separable_conv2d_264/separable_conv2d/depthwise (defined at /Work/InfraredCamera/ThermalView/tests/train_models/predict_mask/train_mask_model_test_old.py:1078) ]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.
 [Op:__inference_train_function_6550581]

Function call stack:
train_function'''




'''DATASET_NAMES_YT = [
    'maskfacesnew',  # not work
    'maskfacesnew3',  # not work
    'maskfacesnewwork_toadd1',  # not work
]

DATASET_NAMES_ALL = [
    'maskfacesnew',  # not work
    'maskfacesnew3',  # not work
    'maskfacesnewwork1',
    'maskfacesnewwork2',
    'maskfacesnewwork_toadd1',  # not work
    'false_faces_work1',
    'source_2020_11_25_1000ms',
    'source_2020_11_25_1000ms5max',
    'source_2020_11_25_200ms',
    'source_2020_11_25_3000ms',
    'source_2020_11_25_3000ms5max',
    'source_2020_11_25_300ms',
    'source_2020_11_25_noms',
    'vk_sources1',  # not work
    # 'worktestset_using_facemasknoses_mask_model_sort',  # 766 files
]

FACE_WIDTH = 128
FACE_HEIGHT = 128
FACE_CHANNELS = 3
PORTION_VAL = 20
USE_WORKTESTSET = False
if USE_WORKTESTSET:
    PORTION_TEST = 0
else:
    PORTION_TEST = 20
EPOCHS = 20
RUNS_PER_EFFECT = 10
EXT_EPOCHS = 200
MAX_CLASS_IMAGES = 999999
SAVE_MODELS = True
SAVE_BEST_LOSS_CHECKPOINTS = True
SAVE_BEST_ACC_CHECKPOINTS = True

results model_best_val_accuracy super test simple loss, test simple acc: [0.15951016545295715, 0.9295774698257446]
Test results at 2021.02.11 00:11
Effect                                                                                               max_val_accuracy  min_val_loss  max_combined  accuracy_simple 
yt-1-20-20-128l256l512l728l1024-128x128x3--gray3 0.926             0.102         1.824         0.92            
all-1-20-20-128l256l512l728l1024-128x128x3--gray3 0.931             0.1           1.821         0.93            
super_max_val_accuracy=0.931      super_max_val_accuracy_key=all-1-20-20-128l256l512l728l1024-128x128x3--gray3 super_min_val_loss=0.1        super_min_val_loss_key=all-1-20-20-128l256l512l728l1024-128x128x3--gray3
super_max_combined=1.824      super_max_combined_key=yt-1-20-20-128l256l512l728l1024-128x128x3--gray3 super_max_combined_max_val_accuracy=0.926      super_max_combined_min_val_loss=0.102     
super_accuracy_simple=0.93       super_accuracy_simple_key=all-1-20-20-128l256l512l728l1024-128x128x3--gray3'''

