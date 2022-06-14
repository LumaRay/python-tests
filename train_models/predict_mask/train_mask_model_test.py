# https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_from_scratch.py
import gc
import math
import os
import time
from random import shuffle

import cv2
import json
import glob
import random
import pathlib
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from multiprocessing import Process
import ctypes as c
import multiprocessing as mp
from multiprocessing.spawn import freeze_support

import copy

# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, Activation, MaxPooling3D, Dense, Dropout, Reshape, LSTM, Flatten, BatchNormalization, Input, concatenate
# USE_PRECISION:int = 16
# from tests.train_models.predict_mask.train_mask_model import PORTION_TEST

USE_PRECISION: int = 32
# USE_PRECISION:int = 64

USE_RANDOM_FLIP = True
# USE_RANDOM_FLIP = False

USE_MULTICORE = True
# USE_MULTICORE = False

# BATCH_SIZE = 32
BATCH_SIZE = 64

# https://graphviz.org/download/
# os.environ['PATH'] = os.environ['PATH']+';'+r"F:\Work\InfraredCamera\graphviz-2.44.1-win32\Graphviz\bin"

OUTPUT_TEST_PATH = "G:/output/tests/"

RESUME_LAST = "resume_last"

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

DATASET_NAMES_ALLW12VK2P1 = [
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
    'v2_source_2020_11_17',
    'v2_source_2020_11_19',
    'v2_source_2020_11_25_1000ms',
    'v2_source_2020_11_25_1000ms5max',
    'v2_source_2020_11_25_200ms',
    'v2_source_2020_11_25_3000ms',
    'v2_source_2020_11_25_3000ms5max',
    'v2_source_2020_11_25_300ms',
    'v2_source_2020_11_25_noms',
    'vk_sources1',  # not work
    'v2_vk_sources2_sorted1',  # not work
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
# SAVE_MODELS_MIN_TEST_ACCURACY = 0.850
SAVE_MODELS_MIN_TEST_ACCURACY = 0

# SAVE_BEST_LOSS_CHECKPOINTS = False
SAVE_BEST_LOSS_CHECKPOINTS = True
# SAVE_BEST_ACC_CHECKPOINTS = False
SAVE_BEST_ACC_CHECKPOINTS = True

USE_EARLY_STOP_BY_SIGNAL = True  #  "early_stop_signal"
early_stopped = False

# ADD_CHANNELS = False
# ADD_CHANNELS = True

# EXTEND_DATASET = False
# EXTEND_DATASET = True

#in_shape = (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS)

SHAPE_APPLY = "sa"
SHAPE_NONE = "sn"
SHAPE_BOTH = "sb"

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
EFFECT_SHAPE = "p"
EFFECT_FILL_GRAY = "gg"
EFFECT_SWAP_CHANNELS = "sw"  #  0 = ABC, 1 = CBA, 2 = BAC, 3 = ACB, 4 = CAB, 5 = BCA

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

EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_FLIP_LIST = [
    [[(EFFECT_NONE, "")]],
    [[(EFFECT_FLIP, "")]],
    [[(EFFECT_TEST1, "")]],
    [[(EFFECT_TEST1, ""), (EFFECT_FLIP, "")]],
    [[(EFFECT_GRAY, 3)]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, "")]],
    [[(EFFECT_BLUR, 3)]],
    [[(EFFECT_BLUR, 3), (EFFECT_FLIP, "")]],
    [[(EFFECT_SHARPEN, 9)]],
    [[(EFFECT_SHARPEN, 9), (EFFECT_FLIP, "")]],
    [[(EFFECT_BRIGHTEN, 10)]],
    [[(EFFECT_BRIGHTEN, 10), (EFFECT_FLIP, "")]],
    [[(EFFECT_DARKEN, 10)]],
    [[(EFFECT_DARKEN, 10), (EFFECT_FLIP, "")]],
    [[(EFFECT_CONTRAST_INC, 10)]],
    [[(EFFECT_CONTRAST_INC, 10), (EFFECT_FLIP, "")]],
    [[(EFFECT_CONTRAST_DEC, 10)]],
    [[(EFFECT_CONTRAST_DEC, 10), (EFFECT_FLIP, "")]],
    [[(EFFECT_NORMALIZE, "")]],
    [[(EFFECT_NORMALIZE, ""), (EFFECT_FLIP, "")]],
    [[(EFFECT_DEVIATION, "")]],
    [[(EFFECT_DEVIATION, ""), (EFFECT_FLIP, "")]],
    [[(EFFECT_DEVIATION2, "")]],
    [[(EFFECT_DEVIATION2, ""), (EFFECT_FLIP, "")]],
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

EXTEND_DATASET_EFFECT_SEQUENCES_GRAY_FLIP_LIST = [
    # [[(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 6)]],
    [[(EFFECT_GRAY, 3)]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, "")]],
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

EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_GRAY_FLIP_LIST = [
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
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_ADAPTIVE, 3)],
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
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_SHARPEN, 9)],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_SHARPEN, 9), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_SHARPEN, 9), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_SHARPEN, 9), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_SHARPEN, 9), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_SHARPEN, 9), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_SHARPEN, 9), (EFFECT_ADAPTIVE, 3)],
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
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_BRIGHTEN, 10)],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_BRIGHTEN, 10), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_BRIGHTEN, 10), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_BRIGHTEN, 10), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_BRIGHTEN, 10), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_BRIGHTEN, 10), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_BRIGHTEN, 10), (EFFECT_ADAPTIVE, 3)],
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
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DARKEN, 10)],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DARKEN, 10), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DARKEN, 10), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DARKEN, 10), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DARKEN, 10), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DARKEN, 10), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DARKEN, 10), (EFFECT_ADAPTIVE, 3)],
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
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_INC, 10)],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_INC, 10), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_INC, 10), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_INC, 10), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_INC, 10), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_INC, 10), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_INC, 10), (EFFECT_ADAPTIVE, 3)],
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
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_DEC, 10)],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_DEC, 10), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_DEC, 10), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_DEC, 10), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_DEC, 10), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_DEC, 10), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_DEC, 10), (EFFECT_ADAPTIVE, 3)],
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
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_NORMALIZE, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_NORMALIZE, ""), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_NORMALIZE, ""), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_NORMALIZE, ""), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_NORMALIZE, ""), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_NORMALIZE, ""), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_NORMALIZE, ""), (EFFECT_ADAPTIVE, 3)],
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
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION, ""), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION, ""), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION, ""), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION, ""), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION, ""), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION, ""), (EFFECT_ADAPTIVE, 3)],
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
    [
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION2, ""), (EFFECT_HSV, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION2, ""), (EFFECT_YCC, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION2, ""), (EFFECT_LAB, 3)],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION2, ""), (EFFECT_TEST1, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION2, ""), (EFFECT_TEST2, "")],
        [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION2, ""), (EFFECT_ADAPTIVE, 3)],
    ],
]

'''EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_GRAY_FLIP_LIST = [
    # [[(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 6)]],
    [[(EFFECT_NONE, 3)]],
    [[(EFFECT_FLIP, "")]],
    [[(EFFECT_TEST1, 3)]],
    [[(EFFECT_TEST1, 3), (EFFECT_FLIP, "")]],
    [[(EFFECT_GRAY, 3)]],
    [[(EFFECT_GRAY, 3), (EFFECT_FLIP, "")]],
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
]'''

EFFECT_PERMUTATIONS_LIST = [
    (EFFECT_NONE, ""),
    (EFFECT_TEST1, ""),
    (EFFECT_GRAY, 3),
    (EFFECT_BLUR, 3),
    (EFFECT_SHARPEN, 9),
    (EFFECT_BRIGHTEN, 10),
    (EFFECT_DARKEN, 10),
    (EFFECT_CONTRAST_INC, 10),
    (EFFECT_CONTRAST_DEC, 10),
    (EFFECT_NORMALIZE, ""),
    (EFFECT_DEVIATION, ""),
    (EFFECT_DEVIATION2, ""),
    (EFFECT_FILL_GRAY, ""),
    (EFFECT_SWAP_CHANNELS, 1),
    (EFFECT_SWAP_CHANNELS, 2),
    (EFFECT_SWAP_CHANNELS, 3),
    # (EFFECT_SWAP_CHANNELS, 4),
    # (EFFECT_SWAP_CHANNELS, 5),
]

'''if ADD_CHANNELS:
    FACE_CHANNELS *= 1 + len(ADD_CHANNELS_EFFECT_SEQUENCES_LIST)'''

MODEL_TUNE_DEFAULT = {
    "simple_conv2d": False,
    "scale_features": 1,
    "add_dense": [],
    "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS),
    "conv_pyramid": [128, 256, 512, 728, 1024],
}

MODEL_TUNE_SIMPLE_CONV2D = {
    "simple_conv2d": True,
    "scale_features": 1,
    "add_dense": [],
    "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS),
    "conv_pyramid": [128, 256, 512, 728, 1024],
}

MODEL_TUNE_SCALE2 = {
    "simple_conv2d": False,
    "scale_features": 2,
    "add_dense": [],
    "in_shape": (1, FACE_WIDTH, FACE_HEIGHT, FACE_CHANNELS),
    "conv_pyramid": [128, 256, 512, 728, 1024]
}

MODEL_TUNE_SCALE2_64X64 = {
    "simple_conv2d": False,
    "scale_features": 2,
    "add_dense": [],
    "in_shape": (1, 64, 64, FACE_CHANNELS),
    "conv_pyramid": [128, 256, 512, 728, 1024]
}

def createEffectExtensionsListFromPermutationsList(effect_permutations_list):
    extensions_list = []
    for inner_effect_type, inner_effect_value in effect_permutations_list:
        if inner_effect_type == EFFECT_FILL_GRAY:
            continue
        for outer_effect_type, outer_effect_value in effect_permutations_list:
            effects_list = [((inner_effect_type, inner_effect_value), (outer_effect_type, outer_effect_value))]
            expansions_list = [effects_list]
            extensions_list.append(expansions_list)
    return extensions_list

def createRandomEffectExtensionsListFromPermutationsWithEffects(effect_permutations_list, effects_depth, variations_count):
    extensions_list = []
    for _ in range(variations_count):
        effects_list = []
        for _ in range(effects_depth):
            inner_effect_type, inner_effect_value = random.choice(effect_permutations_list)
            outer_effect_type, outer_effect_value = random.choice(effect_permutations_list)
            if inner_effect_type == EFFECT_FILL_GRAY:
                continue
            effects_list += [((inner_effect_type, inner_effect_value), (outer_effect_type, outer_effect_value))]
        expansions_list = [effects_list]
        extensions_list.append(expansions_list)
    return extensions_list

def createMultipleTasksWithRandomEffectExtensionsListFromPermutationsWithEffects(effect_permutations_list, effects_depth, variations_count, tasks_count):
    tasks_list = []
    for _ in range(tasks_count):
        task = {
            "effects": createRandomEffectExtensionsListFromPermutationsWithEffects(effect_permutations_list, effects_depth, variations_count),
            # "dataset_names": DATASET_NAMES_ALL,
            "dataset_names": DATASET_NAMES_ALLW12VK2P1,
            # "sub_name": f"allrandd{effects_depth}s{samples_count}t{tasks_count}",
            "sub_name": f"allw12vk2p1randd-{effects_depth}v{variations_count}",  # 5, 20
            "options": {  #  x4 total increase 840 total epochs
                "resume": RESUME_LAST,#None,
                "valid_proportion": 90,  # PORTION_VAL,
                "test_proportion": PORTION_TEST,
                "shape_effect": SHAPE_NONE,
                "runs": 1,#10,
                "epochs": 2,#20,
                "ext_epochs": 0,
            },
            "model_tune": MODEL_TUNE_SIMPLE_CONV2D,
            # "model_tune": MODEL_TUNE_DEFAULT,
        }
        tasks_list.append(task)
    return tasks_list

def extendEffectsListWithFlip(extensions_list):
    append_extensions_list = copy.deepcopy(extensions_list)
    for expansions_list in append_extensions_list:
        for effects_list in expansions_list:
            effects_list.append((EFFECT_FLIP, ""))
    extensions_list += append_extensions_list
    return extensions_list

def repeatListAsChild(a_list, count):
    parent_list = []
    for _ in range(count):
        parent_list += [copy.deepcopy(a_list)]
    return parent_list

TASK_OPTIONS_DEFAULT = {  #  220 total epochs
    "resume": None,
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "shape_effect": SHAPE_APPLY,
    "runs": 10,
    "epochs": 20,
    "ext_epochs": 200,
}

TASK_OPTIONS_EPOCHS_1x1600x0 = {  #  1600 total epochs
    "resume": None,
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "shape_effect": SHAPE_APPLY,
    "runs": 1,
    "epochs": 1600,
    "ext_epochs": 0,
}

TASK_OPTIONS_EPOCHS_1x800x800 = {  #  1600 total epochs
    "resume": None,
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "shape_effect": SHAPE_APPLY,
    "runs": 1,
    "epochs": 800,
    "ext_epochs": 800,
}

TASK_OPTIONS_EPOCHS_x20x40x800 = {  #  x4 total increase 840 total epochs
    "resume": None,
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "shape_effect": SHAPE_APPLY,
    "runs": 10 * 2,
    "epochs": 20 * 2,
    "ext_epochs": 200 * 4,
}

TASK_OPTIONS_EPOCHS_x10x40x400 = {  #  x4 total increase 840 total epochs
    "resume": None,
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "shape_effect": SHAPE_APPLY,
    "runs": 10,
    "epochs": 20 * 2,
    "ext_epochs": 200 * 2,
}

TASK_OPTIONS_EPOCHS_x10x40x200 = {  #
    "resume": None,
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "shape_effect": SHAPE_APPLY,
    "runs": 10,
    "epochs": 20 * 2,
    "ext_epochs": 200,
}

TASK_OPTIONS_SHAPENONE_EPOCHS_x10x40x200 = {  #  x4 total increase 840 total epochs
    "resume": None,
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "shape_effect": SHAPE_NONE,
    "runs": 10,
    "epochs": 20 * 2,
    "ext_epochs": 200,
}

TASK_OPTIONS_SHAPEBOTH_EPOCHS_x10x40x200 = {  #  x4 total increase 840 total epochs
    "resume": None,
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "shape_effect": SHAPE_BOTH,
    "runs": 10,
    "epochs": 20 * 2,
    "ext_epochs": 200,
}

TASK_OPTIONS_EPOCHS_x40x80x800 = {  #  x10 total increase 840 total epochs
    "resume": None,
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "shape_effect": SHAPE_APPLY,
    "runs": 10 * 4,
    "epochs": 20 * 4,
    "ext_epochs": 200 * 4,
}

TASK_OPTIONS_EPOCHS_x20x40x1600 = {  #  x6 total increase 1240 total epochs
    "resume": None,
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "shape_effect": SHAPE_APPLY,
    "runs": 10 * 2,
    "epochs": 20 * 2,
    "ext_epochs": 200 * 8,
}

TASK_OPTIONS_EPOCHS_x20x40x3200 = {  #  x6 total increase 2040 total epochs
    "resume": None,
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    "shape_effect": SHAPE_APPLY,
    "runs": 10 * 2,
    "epochs": 20 * 2,
    "ext_epochs": 200 * 16,
}

TEST_TASKS_LIST_SIMPLE = [{
    "effects": extendEffectsListWithFlip([
        # [[(EFFECT_NONE, "")]]
        [[((EFFECT_NONE, ""), (EFFECT_GRAY, 3))]],
        [[((EFFECT_GRAY, 3), (EFFECT_NONE, ""))]]
    ]), "options": {
    "valid_proportion": PORTION_VAL,
    "test_proportion": PORTION_TEST,
    # "shape_effect": SHAPE_APPLY,
    # "shape_effect": SHAPE_BOTH,
    "shape_effect": SHAPE_NONE,
    "runs": 2,
    # "epochs": 2,
    "epochs": 20,
    "ext_epochs": 10,
}, "model_tune": {
    "scale_features": 1,
    "simple_conv2d": True,
    "add_dense": [],
    "in_shape": (1, 64, 64, FACE_CHANNELS),
    "conv_pyramid": [128, 256, 512, 728, 1024],
}, "dataset_names": DATASET_NAMES_TINY}]
# }, "dataset_names": DATASET_NAMES_SMALL}]
# }, "dataset_names": DATASET_NAMES_WORK}]

# TEST_TASKS_LIST_RANDOM = createMultipleTasksWithRandomEffectExtensionsListFromPermutationsWithEffects(EFFECT_PERMUTATIONS_LIST, 3, 10, 10)
# TEST_TASKS_LIST_RANDOM = createMultipleTasksWithRandomEffectExtensionsListFromPermutationsWithEffects(EFFECT_PERMUTATIONS_LIST, 3, 10, 100)
# TEST_TASKS_LIST_RANDOM = createMultipleTasksWithRandomEffectExtensionsListFromPermutationsWithEffects(EFFECT_PERMUTATIONS_LIST, 5, 20, 1000)
TEST_TASKS_LIST_RANDOM = createMultipleTasksWithRandomEffectExtensionsListFromPermutationsWithEffects(EFFECT_PERMUTATIONS_LIST, 5, 20, 1000)

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
    # {"effects": EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_GRAY_LIST, "model_tune": MODEL_TUNE_SCALE2, "sub_name": "extaddgray"},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_LIST, "options": TASK_OPTIONS_EPOCHS_x4x4x4},
    # {"effects": [[[(EFFECT_GRAY, 3)]], [[(EFFECT_GRAY, 3)]]], "options": TASK_OPTIONS_EPOCHS_x2x2x4},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_LIST, "dataset_names": DATASET_NAMES_ALL, "sub_name": "all"},  #  MemoryError: Unable to allocate 10.9 GiB for an array with shape (238900, 128, 128, 3) and data type uint8
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_FLIP_LIST, "model_tune": MODEL_TUNE_SCALE2_64X64, "options": TASK_OPTIONS_EPOCHS_x10x40x400},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_LIST, "dataset_names": DATASET_NAMES_WORKv12, "sub_name": "work12"},
    # {"effects": EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_FLIP_LIST, "dataset_names": DATASET_NAMES_ALL, "model_tune": MODEL_TUNE_SCALE2, "sub_name": "allextadd"},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_LIST, "options": TASK_OPTIONS_EPOCHS_x20x40x3200},
    # {"effects": EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_LIST, "model_tune": MODEL_TUNE_SCALE2, "sub_name": "extadd1"},
    # {"effects": EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_GRAY_LIST, "model_tune": MODEL_TUNE_SCALE2, "sub_name": "extaddgray1"},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_FLIP_LIST, "dataset_names": DATASET_NAMES_ALL, "model_tune": MODEL_TUNE_SCALE2_64X64, "sub_name": "all", "options": TASK_OPTIONS_EPOCHS_x10x40x400},
    # {"effects": EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_FLIP_LIST, "dataset_names": DATASET_NAMES_ALL, "model_tune": MODEL_TUNE_SCALE2_64X64, "sub_name": "allextaddflip64", "options": TASK_OPTIONS_EPOCHS_x10x40x400},
    # {"effects": EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_GRAY_FLIP_LIST, "dataset_names": DATASET_NAMES_ALL, "model_tune": MODEL_TUNE_SCALE2_64X64, "sub_name": "allextaddgray1", "options": TASK_OPTIONS_EPOCHS_x10x40x400},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_FLIP_LIST, "dataset_names": DATASET_NAMES_ALL, "sub_name": "all", "options": TASK_OPTIONS_SHAPENONE_EPOCHS_x10x40x200},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_GRAY_FLIP_LIST, "dataset_names": DATASET_NAMES_ALL, "sub_name": "all", "options": TASK_OPTIONS_EPOCHS_x10x40x200},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_GRAY_FLIP_LIST, "dataset_names": DATASET_NAMES_ALL, "model_tune": MODEL_TUNE_SCALE2_64X64, "sub_name": "all", "options": TASK_OPTIONS_EPOCHS_x10x40x200},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_FLIP_LIST, "dataset_names": DATASET_NAMES_ALLW12VK2P1, "sub_name": "allw12vk2p1", "options": TASK_OPTIONS_EPOCHS_x10x40x200},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_FLIP_LIST, "dataset_names": DATASET_NAMES_ALLW12VK2P1, "sub_name": "allw12vk2p1", "options": TASK_OPTIONS_SHAPENONE_EPOCHS_x10x40x200},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_FLIP_LIST, "dataset_names": DATASET_NAMES_ALL, "sub_name": "all", "options": TASK_OPTIONS_SHAPEBOTH_EPOCHS_x10x40x200},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_FLIP_LIST, "dataset_names": DATASET_NAMES_ALLW12VK2P1, "model_tune": MODEL_TUNE_SCALE2, "sub_name": "allw12vk2p1", "options": TASK_OPTIONS_EPOCHS_x10x40x400},
    {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_FLIP_LIST, "dataset_names": DATASET_NAMES_ALLW12VK2P1, "sub_name": "allw12vk2p1", "options": TASK_OPTIONS_EPOCHS_x20x40x800},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_FLIP_LIST, "dataset_names": DATASET_NAMES_ALL, "model_tune": {"scale_features": 1, "add_dense": [], "in_shape": (1, 256, 256, FACE_CHANNELS), "conv_pyramid": [128, 256, 512, 728, 1024]}, "sub_name": "all", "options": TASK_OPTIONS_EPOCHS_x20x40x800},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_FLIP_LIST, "dataset_names": DATASET_NAMES_ALL, "model_tune": MODEL_TUNE_SCALE2, "sub_name": "all", "options": TASK_OPTIONS_EPOCHS_x20x40x800},
    # {"effects": EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_GRAY_FLIP_LIST, "model_tune": MODEL_TUNE_SCALE2, "sub_name": "extaddgray1", "options": TASK_OPTIONS_EPOCHS_x10x40x400},
    # {"effects": EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_FLIP_LIST, "options": TASK_OPTIONS_EPOCHS_x20x40x3200},
    # {"effects": EXTEND_DATASET_ADD_CHANNELS_EFFECT_SEQUENCES_GRAY_FLIP_LIST, "dataset_names": DATASET_NAMES_ALL, "model_tune": MODEL_TUNE_SCALE2, "sub_name": "allextaddgray1", "options": TASK_OPTIONS_EPOCHS_x20x40x800},
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

parsed_data_path_set = []
for root, subdirs, files in os.walk(parsed_data_path):
    files_set = set(files)
    parsed_data_path_set.append((root, files_set))

def findParsedFile(file_name, r):
    # global parsed_data_path  #  , parsed_data_last_path_list
    for parsed_data_last_path in r["parsed_data_last_path_list"]:
        check_path = os.path.join(parsed_data_last_path, file_name)
        if os.path.isfile(check_path):
            return check_path
    # for root, subdirs, files in os.walk(parsed_data_path):
    #     if file_name in files:
    for root, files_set in parsed_data_path_set:
        if file_name in files_set:
            parsed_data_last_path_list = r["parsed_data_last_path_list"]
            parsed_data_last_path_list.insert(0, root)
            r["parsed_data_last_path_list"] = parsed_data_last_path_list
            return os.path.join(root, file_name)
    return None

def applyEffectsList(img, effects_list, contour_scaled):
    for effect_left, effect_right in effects_list:
        if effect_left is not None and isinstance(effect_left, str):
            effect_type, effect_value = effect_left, effect_right
            img = applyEffect(img, effect_type, effect_value, contour_scaled)
        elif effect_left is None:
            img = None
        else:
            inner_effect_type, inner_effect_value = effect_left
            outer_effect_type, outer_effect_value = effect_right
            img = applyMaskEffectPair(img, inner_effect_type, inner_effect_value, outer_effect_type, outer_effect_value, contour_scaled)
    return img

def applyEffect(img, effect_type, effect_value, contour_scaled):
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
    elif effect_type == EFFECT_SWAP_CHANNELS:
        # 0 = ABC, 1 = CBA, 2 = BAC, 3 = ACB, 4 = CAB, 5 = BCA
        if effect_value == 0:
            pass
        elif effect_value == 1:
            img = img[:, :, (2, 1, 0)]
        elif effect_value == 2:
            img = img[:, :, (1, 0, 2)]
        elif effect_value == 3:
            img = img[:, :, (0, 2, 1)]
        elif effect_value == 4:
            img = img[:, :, (2, 0, 1)]
        elif effect_value == 5:
            img = img[:, :, (1, 2, 0)]
    elif effect_type == EFFECT_FILL_GRAY:
        img.fill(127)
    elif effect_type == EFFECT_FLIP:
        img = cv2.flip(img, 1)
    elif effect_type == EFFECT_SHAPE:
        mask_roi = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask_roi, [contour_scaled], -1, (255), thickness=-1)
        res_roi = cv2.bitwise_and(img, img, mask=mask_roi)
        mask_roi = 255 - mask_roi
        gray_roi = np.zeros(img.shape, np.uint8)
        gray_roi.fill(127)
        gray_roi_reversed = cv2.bitwise_and(gray_roi, gray_roi, mask=mask_roi)
        img = cv2.addWeighted(res_roi, 1, gray_roi_reversed, 1, 0)

    ''' if (effect_type == EFFECT_IN_SHAPE) and (effect_value[3] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)'''

    return img

def applyMaskEffectPair(img, inner_effect_type, inner_effect_value, outer_effect_type, outer_effect_value, contour_scaled):
    inner_img = applyEffect(img, inner_effect_type, inner_effect_value, contour_scaled)
    inner_mask_roi = np.zeros(inner_img.shape[:2], np.uint8)
    cv2.drawContours(inner_mask_roi, [contour_scaled], -1, (255), thickness=-1)
    inner_res_roi = cv2.bitwise_and(inner_img, inner_img, mask=inner_mask_roi)
    outer_img = applyEffect(img, outer_effect_type, outer_effect_value, contour_scaled)
    outer_mask_roi = 255 - inner_mask_roi
    outer_res_roi = cv2.bitwise_and(outer_img, outer_img, mask=outer_mask_roi)
    img = cv2.addWeighted(inner_res_roi, 1, outer_res_roi, 1, 0)
    return img

def loadDatasetSourceList(dataset_names, r):
    res_data_list = []
    classes_source_path_list = [dataset_sources_path + '/' + dsn for dsn in dataset_names]
    for class_idx, class_name in enumerate(CLASS_NAMES):
        for classes_source_path in classes_source_path_list:
            class_path = classes_source_path + '/' + class_name
            files_list = os.listdir(class_path)
            print("Parsing class " + class_name + " of size " + str(len(files_list)) + " with effect " + r["effect_print"] + " from " + class_path)
            new_class_idx = CLASSES_MAP[class_idx]
            new_class_name = CLASS_NAMES[new_class_idx]
            print("Mapping class {} to {}".format(class_name, new_class_name))
            final_class_idx = class_indexes.index(new_class_idx)
            for fidx, file in enumerate(files_list):
                if fidx > MAX_CLASS_IMAGES:
                    break
                if file.endswith(".jpg"):
                    file_name, file_ext = file.split('.')
                    json_object_path = findParsedFile(file_name + '.txt', r)
                    with open(json_object_path) as json_file:
                        json_entry = json.load(json_file)
                        frame_ref = json_entry['frame_ref']
                        frame_path = findParsedFile(frame_ref + ".jpg", r)
                        if os.path.isfile(frame_path):
                            res_data_list.append((json_entry, frame_path, final_class_idx))
    return res_data_list

def loadClassesToDatasets(options, dataset_source_list, extend_dataset_list, model_tune, mp_X_arr, dataset_length, input_size, mp_y_arr, output_size, cpu_idx, cpu_count):  #  , add_channels, extend_dataset):
    # global effect_print

    face_width = model_tune["in_shape"][1]
    face_height = model_tune["in_shape"][2]
    face_channels = model_tune["in_shape"][3]
    face_channels *= len(extend_dataset_list[0])

    x_out = np.frombuffer(mp_X_arr.get_obj(), dtype=np.uint8)
    # x_out = x_out.reshape((dataset_length, face_height, face_width, face_channels * len(extend_dataset_list[0])))
    x_out = x_out.reshape((dataset_length, face_height, face_width, face_channels))
    y_out = np.frombuffer(mp_y_arr.get_obj(), dtype=np.float32)
    y_out = y_out.reshape((dataset_length, output_size))

    cpu_samples_count = int(math.floor(len(dataset_source_list) / cpu_count))
    cpu_start_sample_idx = cpu_idx * cpu_samples_count
    if cpu_idx < cpu_count - 1:
        cpu_end_sample_idx = cpu_start_sample_idx + cpu_samples_count
    else:
        cpu_end_sample_idx = len(dataset_source_list)
    last_frame_path = ""
    last_frame_img = None
    for sample_idx in range(cpu_start_sample_idx, cpu_end_sample_idx):
        '''sample_idx = sample_idx * cpu_count + cpu_idx
        if sample_idx >= len(dataset_source_list):
            break'''
        dataset_source = dataset_source_list[sample_idx]
        json_entry, frame_path, final_class_idx = dataset_source
        # frame_ref = json_entry['frame_ref']
        bbox = json_entry['bbox']
        bbox_x = bbox['x']
        bbox_y = bbox['y']
        bbox_w = bbox['w']
        bbox_h = bbox['h']
        contour = json_entry['contour']
        np_contour = np.zeros((len(contour), 2), dtype=np.int32)
        for i, contour_point in enumerate(contour):
            x = contour_point['x'] - bbox_x
            y = contour_point['y'] - bbox_y
            np_contour[i] = [x, y]
        # frame_path = findParsedFile(frame_ref + ".jpg")
        # try:
        if (frame_path == last_frame_path) and (last_frame_img is not None):
            img = last_frame_img.copy()
        else:
            img = cv2.imread(frame_path)
            last_frame_img = img.copy()
            last_frame_path = frame_path
        img_roi = img[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]

        np_contour_scaled = (np_contour * [face_width / img_roi.shape[1], face_height / img_roi.shape[0]]).astype(np.int32)

        if (options["shape_effect"] == SHAPE_APPLY) or (options["shape_effect"] == SHAPE_BOTH):
            mask_roi = np.zeros(img_roi.shape[:2], np.uint8)
            cv2.drawContours(mask_roi, [np_contour], -1, (255), thickness=-1)
            res_roi = cv2.bitwise_and(img_roi, img_roi, mask=mask_roi)
            mask_roi = 255 - mask_roi
            gray_roi = np.zeros(img_roi.shape, np.uint8)
            gray_roi.fill(127)
            gray_roi_reversed = cv2.bitwise_and(gray_roi, gray_roi, mask=mask_roi)
            shape_roi = cv2.addWeighted(res_roi, 1, gray_roi_reversed, 1, 0)
            shape_roi = cv2.resize(shape_roi, (face_width, face_height), cv2.INTER_CUBIC)
            res_shape_roi_original = shape_roi

        if options["shape_effect"] == SHAPE_APPLY:
            img_roi = shape_roi

        res_roi = img_roi
        res_roi = cv2.resize(res_roi, (face_width, face_height), cv2.INTER_CUBIC)
        res_roi_original = res_roi
        for extend_idx, expand_dataset_list in enumerate(extend_dataset_list):
            res_roi = None
            for effects_list in expand_dataset_list:
                add_to_res_roi = res_roi_original.copy()
                add_to_res_roi = applyEffectsList(add_to_res_roi, effects_list, np_contour_scaled)
                if res_roi is None:
                    res_roi = add_to_res_roi
                else:
                    res_roi = np.concatenate((res_roi, add_to_res_roi), axis=2)
            '''if USE_PRECISION == int(16):
                X[sample_idx] = res_roi.astype(np.float16)
                y[sample_idx] = np.zeros(num_classes).astype(np.float16)
            if USE_PRECISION == int(64):
                X[sample_idx] = res_roi.astype(np.float64)
                y[sample_idx] = np.zeros(num_classes).astype(np.float64)
            else:
                X[sample_idx] = res_roi.astype(np.float32)
                y[sample_idx] = np.zeros(num_classes).astype(np.float32)'''
            extended_sample_idx = sample_idx * len(extend_dataset_list) + extend_idx
            x_out[extended_sample_idx] = res_roi.astype(np.uint8)
            y_out[extended_sample_idx] = np.zeros(num_classes).astype(np.float32)
            y_out[extended_sample_idx][final_class_idx] = 1

            if options["shape_effect"] == SHAPE_BOTH:
                res_roi = None
                for effects_list in expand_dataset_list:
                    add_to_res_roi = res_shape_roi_original.copy()
                    add_to_res_roi = applyEffectsList(add_to_res_roi, effects_list, np_contour_scaled)
                    if res_roi is None:
                        res_roi = add_to_res_roi
                    else:
                        res_roi = np.concatenate((res_roi, add_to_res_roi), axis=2)
                extended_sample_idx = int(len(x_out) / 2) + sample_idx * len(extend_dataset_list) + extend_idx
                x_out[extended_sample_idx] = res_roi.astype(np.uint8)
                y_out[extended_sample_idx] = np.zeros(num_classes).astype(np.float32)
                y_out[extended_sample_idx][final_class_idx] = 1
        # except:
        #     print("Failed to process " + frame_path + " !!!")

    # return (lst_x_dataset, lst_y_dataset)

    #x_train = x_train.map(lambda x, y: (data_augmentation(x, training=True), y))
    # x_dataset, y_dataset = np.array(lst_x_dataset), np.array(lst_y_dataset)
    # lst_x_dataset, lst_y_dataset = [], []
    # gc.collect()
    # if USE_PRECISION == int(16):
    #     x_dataset, y_dataset = x_dataset.astype(np.float16), y_dataset.astype(np.float16)
    # elif USE_PRECISION == int(64):
    #     x_dataset, y_dataset = x_dataset.astype(np.float64), y_dataset.astype(np.float64)
    # gc.collect()
    # return (x_dataset, y_dataset)
    #tmp = list(zip(lst_x_dataset, lst_y_dataset))
    #random.shuffle(tmp)
    #lst_x_dataset, lst_y_dataset = zip(*tmp)
    #return (np.array(lst_x_dataset), np.array(lst_y_dataset))

''' BEFORE 10.12.21
def make_model(input_shape, num_classes, add_dense=[], features_scale_factor=1, conv_pyramid=[128, 256, 512, 728, 1024]):
    from tensorflow.keras import layers
    from tensorflow import keras
    if USE_RANDOM_FLIP:
        data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                layers.experimental.preprocessing.RandomRotation(0.1),
                # layers.experimental.preprocessing.RandomWidth(0.1, "cubic"),
                # layers.experimental.preprocessing.RandomContrast(0.2)
            ]
        )
    else:
        data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomRotation(0.1),
            ]
        )

    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    #x = inputs
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)  # GET RID OF THIS = ERRORS ON CONVERSION !!!!
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
        x = layers.Activation("relu")(x)  # ACTIVATE FIRST CYCLE AGAIN????
        x = layers.SeparableConv2D(size, 3, padding="same")(x)  # Try replacing with simple CONV2D strides=2 or no strides param
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)  # MOVE TO TOP OF THE CYCLE???
        # residual: NO NORMALIZATION???
        x = layers.add([x, residual])  # Add back residual
        # ACTIVATION SHOULD BE HERE OR LAST CYCLE IS NOT ACTIVATED!!!!
        previous_block_activation = x  # Set aside next residual

    # x = layers.SeparableConv2D(round(1024 * features_scale_factor), 3, padding="same")(x)
    x = layers.SeparableConv2D(round(conv_pyramid[-1] * features_scale_factor), 3, padding="same")(x)  # NO ACTIVATION BEFORE THIS???
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    # if num_classes == 2:
    #     activation = "sigmoid"
    #     units = 1
    # else:
    activation = "softmax"
    units = num_classes

    x = layers.Dropout(0.5)(x)
    for dense_size in add_dense:
        x = layers.Dense(dense_size, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)'''

def make_model(input_shape, num_classes, add_dense=[], features_scale_factor=1, conv_pyramid=[128, 256, 512, 728, 1024], use_simple_conv2d=False):
    from tensorflow.keras import layers
    from tensorflow import keras
    if USE_RANDOM_FLIP:
        data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                layers.experimental.preprocessing.RandomRotation(0.1),
                # layers.experimental.preprocessing.RandomWidth(0.1, "cubic"),
                # layers.experimental.preprocessing.RandomContrast(0.2)
            ]
        )
    else:
        data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomRotation(0.1),
            ]
        )

    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    #x = inputs
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)  # GET RID OF THIS = ERRORS ON CONVERSION !!!!
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
        if use_simple_conv2d:
            x = layers.Conv2D(size, 3, padding="same")(x)
        else:
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        if use_simple_conv2d:
            x = layers.Conv2D(size, 3, padding="same")(x)
        else:
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        residual = layers.BatchNormalization()(residual)
        # ADD MAX POOLING???
        x = layers.add([x, residual])  # Add back residual
        x = layers.Activation("relu")(x)
        previous_block_activation = x  # Set aside next residual

    # x = layers.SeparableConv2D(round(1024 * features_scale_factor), 3, padding="same")(x)
    if use_simple_conv2d:
        x = layers.Conv2D(round(conv_pyramid[-1] * features_scale_factor), 3, padding="same")(x)
    else:
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

def printTestResults(min_val_loss, max_val_accuracy, accuracy_simple, r):
    # global test_results, super_max_val_accuracy, super_max_val_accuracy_key, super_min_val_loss, super_min_val_loss_key, super_max_combined, super_max_combined_key, super_accuracy_simple, super_accuracy_simple_key
    print("Test results at " + datetime.now().strftime("%Y.%m.%d %H:%M"))
    # test_results
    # print(test_results)
    print("{:<100} {:<17} {:<13} {:<13} {:<16}".format('Effect', 'max_val_accuracy', 'min_val_loss', 'max_combined', 'accuracy_simple'))
    for k, v in r["test_results"].items():
        if v["max_val_accuracy"] > r["super_max_val_accuracy"]:
            r["super_max_val_accuracy"] = v["max_val_accuracy"]
            r["super_max_val_accuracy_key"] = k
        if v["min_val_loss"] < r["super_min_val_loss"]:
            r["super_min_val_loss"] = v["min_val_loss"]
            r["super_min_val_loss_key"] = k
        if v["max_combined"] > r["super_max_combined"]:
            r["super_max_combined"] = v["max_combined"]
            r["super_max_combined_key"] = k
        if v["accuracy_simple"] > r["super_accuracy_simple"]:
            r["super_accuracy_simple"] = v["accuracy_simple"]
            r["super_accuracy_simple_key"] = k
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
            round(r["super_max_val_accuracy"], 3), r["super_max_val_accuracy_key"], round(r["super_min_val_loss"], 3), r["super_min_val_loss_key"]))
    print("super_max_combined={:<10} super_max_combined_key={} super_max_combined_max_val_accuracy={:<10} super_max_combined_min_val_loss={:<10}".format(
            round(r["super_max_combined"], 3), r["super_max_combined_key"],
            round(r["test_results"][r["super_max_combined_key"]]["max_val_accuracy"] if r["super_max_combined_key"] != '' else 0, 3),
            round(r["test_results"][r["super_max_combined_key"]]["min_val_loss"] if r["super_max_combined_key"] != '' else 1, 3)))
    print("super_accuracy_simple={:<10} super_accuracy_simple_key={}".format(
            round(r["super_accuracy_simple"], 3), r["super_accuracy_simple_key"]))

def formatEffectString(effect_left, effect_right):
    if isinstance(effect_left, str):
        effect_type, effect_value = effect_left, effect_right
        effect_print = effect_type
        if effect_value != "":
            effect_print += str(effect_value)
        return effect_print
    else:
        inner_effect_type, inner_effect_value = effect_left
        outer_effect_type, outer_effect_value = effect_right
        return formatEffectString(inner_effect_type, inner_effect_value) + formatEffectString(outer_effect_type, outer_effect_value) + "i"

# def runTest(effect_type, effect_value):
def taskTest(test_task, output_test_path, r):
    from tensorflow.keras import backend as K

    import tensorflow as tf

    global early_stopped

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if USE_PRECISION == int(16):
        K.set_floatx('float16')
    elif USE_PRECISION == int(64):
        K.set_floatx('float64')

    # global effect_print, test_results

    # tf.config.experimental.set_memory_growth(physical_devices[0], False)
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # MODEL_TUNE_DEFAULT = {"scale_features": 1, "valid_proportion": 20, "add_dense": [], "in_shape": (1, 128, 128, 3), "conv_pyramid": [128, 256, 512, 728, 1024]}

    extend_dataset_list = test_task["effects"]

    sub_name = ""

    if "sub_name" in test_task:
        sub_name = test_task["sub_name"]

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

    face_channels *= len(extend_dataset_list[0])


    add_dense_str = "d".join(str(dense_size) for dense_size in model_tune["add_dense"])
    if add_dense_str != "":
        add_dense_str = "-" + add_dense_str

    conv_pyramid_str = "l".join(str(conv_layer_size) for conv_layer_size in model_tune["conv_pyramid"])
    if conv_pyramid_str != "":
        conv_pyramid_str = "-" + conv_pyramid_str

    effects_str = ""

    for expand_dataset_list in extend_dataset_list:
        effects_str += "-"
        for effects_list in expand_dataset_list:
            effects_str += "-"
            for e_left, e_right in effects_list:
                effects_str += formatEffectString(e_left, e_right)

    if len(effects_str) > 150:
        effects_str = ""


    effect_print_left = "{sub_name}{simple_conv2d}{scale_features}-{valid_proportion}-{test_proportion}{add_dense}{conv_pyramid}".format(
        sub_name=(sub_name + ("-" if sub_name != "" else "")),
        scale_features=model_tune["scale_features"],
        valid_proportion=options["valid_proportion"],
        test_proportion=options["test_proportion"],
        simple_conv2d=('sc-' if model_tune["simple_conv2d"] else ''),
        add_dense=add_dense_str,
        conv_pyramid=conv_pyramid_str,
    )

    effect_print_right = "{face_width}x{face_height}x{face_channels}-{shape_effect}{effects_str}".format(
        face_width=face_width,
        face_height=face_height,
        face_channels=face_channels,
        shape_effect=options["shape_effect"],
        effects_str=effects_str,
    )

    r["effect_print"] = effect_print_left + "-" + effect_print_right

    print("Training test model for " + r["effect_print"])

    # gc.collect()

    # input_size = len(extend_dataset_list[0]) * face_width * face_height * face_channels
    input_size = face_width * face_height * face_channels
    output_size = num_classes

    dataset_source_list = loadDatasetSourceList(dataset_names, r)

    dataset_length = len(dataset_source_list) * len(extend_dataset_list)

    '''if USE_PRECISION == int(16):
        mp_X_arr = mp.Array(c.c_float, len(dataset_source_list) * input_size)
        mp_y_arr = mp.Array(c.c_float, len(dataset_source_list) * output_size)
    elif USE_PRECISION == int(64):
        mp_X_arr = mp.Array(c.c_longdouble, len(dataset_source_list) * input_size)
        mp_y_arr = mp.Array(c.c_longdouble, len(dataset_source_list) * output_size)
    else:
        mp_X_arr = mp.Array(c.c_double, len(dataset_source_list) * input_size)
        mp_y_arr = mp.Array(c.c_double, len(dataset_source_list) * output_size)'''

    if options["shape_effect"] == SHAPE_BOTH:
        dataset_length *= 2

    mp_X_arr = mp.Array(c.c_byte, dataset_length * input_size)
    mp_y_arr = mp.Array(c.c_float, dataset_length * output_size)

    # cpu_count = mp.cpu_count()
    if USE_MULTICORE:
        cpu_count = int(mp.cpu_count() / 2)
    else:
        cpu_count = 1
    print(f"Found {cpu_count} CPUs")
    start = time.time()
    prep_processes = []
    for cpu_idx in range(cpu_count):
        p = mp.Process(target=loadClassesToDatasets, args=(options, dataset_source_list, extend_dataset_list, model_tune, mp_X_arr, dataset_length, input_size, mp_y_arr, output_size, cpu_idx, cpu_count,))
        p.start()
        prep_processes += [p]
    '''for p in prep_processes:
        p.join()'''
    [p.join() for p in prep_processes]
    # map(lambda p: p.join(), prep_processes)
    '''for p in prep_processes:
        p.close()'''
    [p.close() for p in prep_processes]
    # map(lambda p: p.close(), prep_processes)
    end = time.time()
    print(f"Runtime of general dataset prep is {end - start}")
    # loadClassesToDatasets(dataset_source_list, extend_dataset_list, model_tune, mp_X_arr, input_size, mp_y_arr, output_size)  # , add_channels, extend_dataset)

    mp_X_test_simple_arr = None
    mp_y_test_simple_arr = None
    if USE_WORKTESTSET:
        dataset_source_list_test_simple = loadDatasetSourceList(DATASET_NAMES_WORKTESTSET, r)
        '''if USE_PRECISION == int(16):
            mp_X_test_simple_arr = mp.Array(c.c_float, len(dataset_source_list_test_simple) * input_size)
            mp_y_test_simple_arr = mp.Array(c.c_float, len(dataset_source_list_test_simple) * output_size)
        elif USE_PRECISION == int(64):
            mp_X_test_simple_arr = mp.Array(c.c_longdouble, len(dataset_source_list_test_simple) * input_size)
            mp_y_test_simple_arr = mp.Array(c.c_longdouble, len(dataset_source_list_test_simple) * output_size)
        else:
            mp_X_test_simple_arr = mp.Array(c.c_double, len(dataset_source_list_test_simple) * input_size)
            mp_y_test_simple_arr = mp.Array(c.c_double, len(dataset_source_list_test_simple) * output_size)'''

        dataset_test_simple_length = len(dataset_source_list_test_simple) * len(extend_dataset_list)

        if options["shape_effect"] == SHAPE_BOTH:
            dataset_test_simple_length *= 2

        mp_X_test_simple_arr = mp.Array(c.c_byte, dataset_test_simple_length * input_size)
        mp_y_test_simple_arr = mp.Array(c.c_float, dataset_test_simple_length * output_size)

        # x_out_test_simple = np.frombuffer(mp_X_test_simple_arr.get_obj())
        # x_out_test_simple = x_out_test_simple.reshape((len(dataset_source_list_test_simple), input_size))
        # y_out_test_simple = np.frombuffer(mp_y_test_simple_arr.get_obj())
        # y_out_test_simple = y_out_test_simple.reshape((len(dataset_source_list_test_simple), output_size))
        start = time.time()
        prep_processes = []
        for cpu_idx in range(cpu_count):
            p = mp.Process(target=loadClassesToDatasets, args=(options, dataset_source_list_test_simple, extend_dataset_list, model_tune, mp_X_test_simple_arr, dataset_test_simple_length, input_size, mp_y_test_simple_arr, output_size, cpu_idx, cpu_count,))
            p.start()
            prep_processes += [p]
        for p in prep_processes:
            p.join()
        for p in prep_processes:
            p.close()
        end = time.time()
        print(f"Runtime of test dataset prep is {end - start}")
        # loadClassesToDatasets(dataset_source_list, extend_dataset_list, model_tune, mp_X_test_simple_arr, input_size, mp_y_test_simple_arr, output_size)  # add test dataset


    total_runs = options["runs"]
    if options["ext_epochs"] > 0:
        total_runs += 1

    # model_best_simple_accuracy = None

    r["model_best_simple_accuracy_path"] = None
    # global effect_print, test_results,

    for run_idx in range(total_runs):
        # taskRun(model_tune, options, total_runs, run_idx, mp_X_arr, mp_y_arr, mp_X_test_simple_arr, mp_y_test_simple_arr, input_size, output_size)
        p = mp.Process(target=taskRun, args=(model_tune, options, total_runs, run_idx, mp_X_arr, mp_y_arr, mp_X_test_simple_arr, mp_y_test_simple_arr, input_size, output_size, face_channels, output_test_path, r))
        p.start()
        p.join()
        p.close()
        # return_dict["model_best_simple_accuracy_path"]
        del p
        gc.collect()
        if early_stopped:
            break

    del mp_X_arr
    del mp_y_arr
    del mp_X_test_simple_arr
    del mp_y_test_simple_arr


def taskRun(model_tune, options, total_runs, run_idx, mp_X_arr, mp_y_arr, mp_X_test_simple_arr, mp_y_test_simple_arr, input_size, output_size, face_channels, output_test_path, r):
    from tensorflow import keras
    print("||| Run", run_idx + 1, "for test model", r["effect_print"])

    face_width = model_tune["in_shape"][1]
    face_height = model_tune["in_shape"][2]

    input_size = face_height * face_width * face_channels

    samples_count = int(len(mp_X_arr) / input_size)
    x_out = np.frombuffer(mp_X_arr.get_obj(), dtype=np.uint8)
    x_out = x_out.reshape((samples_count, face_height, face_width, face_channels))
    y_out = np.frombuffer(mp_y_arr.get_obj(), dtype=np.float32)
    y_out = y_out.reshape((samples_count, output_size))

    if USE_WORKTESTSET:
        samples_test_simple_count = int(len(mp_X_test_simple_arr) / input_size)
        x_test_simple = np.frombuffer(mp_X_test_simple_arr.get_obj(), dtype=np.uint8)
        x_test_simple = x_test_simple.reshape((samples_test_simple_count, face_height, face_width, face_channels))
        y_test_simple = np.frombuffer(mp_y_test_simple_arr.get_obj(), dtype=np.float32)
        y_test_simple = y_test_simple.reshape((samples_test_simple_count, output_size))

    proportion_val = options["valid_proportion"]

    count_train = int(samples_count * (100 - proportion_val - PORTION_TEST) / 100)
    count_val = int(samples_count * proportion_val / 100)
    count_test = int(samples_count * PORTION_TEST / 100)

    model_name = MODEL_NAME + ("_" if MODEL_NAME != "" else "")  # + sub_name + ("_" if sub_name != "" else "")

    # model_best_simple_accuracy_path = r["model_best_simple_accuracy_path"]

    '''current_test_best_simple_accuracy = 0.1234
    current_test_best_simple_loss = 0.5678
    current_val_accuracy = 0.0123
    current_val_loss = 0.0456
    epochs = 123
    testpath = output_test_path + "/ta{0:03d}".format(int(round(current_test_best_simple_accuracy * 1000, 0))) + "tl{0:03d}".format(int(round(current_test_best_simple_loss * 1000, 0))) + "_i_va{:.3f}".format(round(current_val_accuracy, 3)) + "vl{:.3f}".format(round(current_val_loss, 3)) + "_r{0:02d}".format(run_idx + 1) + "e{0:03d}".format(epochs) + '_' + model_name + effect_print + '.h5'
    print(testpath)'''

    # tmp = list(zip([x for x in x_out], [y for y in y_out]))
    # x_out, y_out = None, None
    # gc.collect()
    # random.shuffle(tmp)
    # lst_x_dataset, lst_y_dataset = zip(*tmp)
    # del tmp
    # tmp = None
    # gc.collect()
    # x_out, y_out = np.array(lst_x_dataset), np.array(lst_y_dataset)
    # del lst_x_dataset
    # del lst_y_dataset
    # lst_x_dataset, lst_y_dataset = None, None

    rng_state = np.random.get_state()
    np.random.shuffle(x_out)
    np.random.set_state(rng_state)
    np.random.shuffle(y_out)

    '''permute_positions_list = list(range(len(x_out)))
    permute_shuffled_list = permute_positions_list.copy()
    shuffle(permute_shuffled_list)
    for sample_idx in range(len(x_out)):
        new_sample_pos = permute_shuffled_list[sample_idx]
        new_sample_pos = permute_positions_list[new_sample_pos]
        x_out[new_sample_pos], x_out[sample_idx] = x_out[sample_idx], x_out[new_sample_pos]
        y_out[new_sample_pos], y_out[sample_idx] = y_out[sample_idx], y_out[new_sample_pos]
        permute_positions_list[sample_idx], permute_positions_list[new_sample_pos] = new_sample_pos, sample_idx'''

    if USE_WORKTESTSET:
        rng_state = np.random.get_state()
        np.random.shuffle(x_test_simple)
        np.random.set_state(rng_state)
        np.random.shuffle(y_test_simple)

    '''if USE_WORKTESTSET:
        permute_positions_list = list(range(len(x_test_simple)))
        permute_shuffled_list = permute_positions_list.copy()
        shuffle(permute_shuffled_list)
        for sample_idx in range(len(x_test_simple)):
            new_sample_pos = permute_shuffled_list[sample_idx]
            new_sample_pos = permute_positions_list[new_sample_pos]
            x_test_simple[new_sample_pos], x_test_simple[sample_idx] = x_test_simple[sample_idx], x_test_simple[new_sample_pos]
            y_test_simple[new_sample_pos], y_test_simple[sample_idx] = y_test_simple[sample_idx], y_test_simple[new_sample_pos]
            permute_positions_list[sample_idx], permute_positions_list[new_sample_pos] = new_sample_pos, sample_idx'''

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
        model = None
        if ("resume" in options) and (options["resume"] == RESUME_LAST):
            list_of_files = glob.glob(output_test_path + '/*.h5')
            if len(list_of_files) > 0:
                latest_file = max(list_of_files, key=os.path.getctime)
                print("Loading to resume: " + latest_file)
                model = keras.models.load_model(latest_file)
        if (model is None) and ("resume" in options) and (options["resume"] is not None) and (options["resume"] != "") and (options["resume"] != RESUME_LAST):
            model = keras.models.load_model(options["resume"])
        if model is None:
            model = make_model(
                input_shape=(face_width, face_height, face_channels),
                num_classes=num_classes,
                add_dense=model_tune["add_dense"],
                features_scale_factor=model_tune["scale_features"],
                conv_pyramid=model_tune["conv_pyramid"],
                use_simple_conv2d=model_tune["simple_conv2d"],
            )
            # keras.utils.plot_model(model, show_shapes=True)
            # opt = tf.train.AdamOptimizer(1e-3, epsilon=1e-4)
            model.compile(
                optimizer=(
                    keras.optimizers.Adam(1e-3) if USE_PRECISION == int(32) else keras.optimizers.Adam(1e-3, epsilon=1e-4)),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
        epochs = options["epochs"]
    else:
        if r["model_best_simple_accuracy_path"] is None:
            return
        model = keras.models.load_model(r["model_best_simple_accuracy_path"])
        # model_best_simple_accuracy_path = None
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

    # time_stamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S_%f")
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M")

    callbacks = []
    # a{accuracy:.2f}l{loss:.2f}
    # str(face_width) + 'x' + str(face_height) + 'x' + str(face_channels) + '_' +
    # "_" + time_stamp +
    if SAVE_MODELS:
        if USE_EARLY_STOP_BY_SIGNAL:
            '''class EarlyStoppingByAccuracy(keras.callbacks.Callback):
                def __init__(self, monitor='accuracy', value=0.98, verbose=0):
                    super(keras.callbacks.Callback, self).__init__()
                    self.monitor = monitor
                    self.value = value
                    self.verbose = verbose

                def on_epoch_end(self, epoch, logs={}):
                    current = logs.get(self.monitor)
                    if current is None:
                        warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

                    if current >= self.value:
                        if self.verbose > 0:
                            print("Epoch %05d: early stopping THR" % epoch)
                        self.model.stop_training = True

            callbacks.append(EarlyStoppingByAccuracy(monitor='accuracy', value=0.98, verbose=1))'''
            class EarlyStoppingBySignal(keras.callbacks.Callback):
                def __init__(self):
                    super(keras.callbacks.Callback, self).__init__()

                def on_epoch_end(self, epoch, logs={}):
                    global early_stopped
                    early_stop_signal_file = pathToScriptFolder + "/early_stop_signal"
                    if os.path.exists(early_stop_signal_file):
                        print("Epoch %05d: early stopping THR" % epoch)
                        self.model.stop_training = True
                        os.remove(early_stop_signal_file)
                        early_stopped = True

            callbacks.append(EarlyStoppingBySignal())
        if SAVE_BEST_LOSS_CHECKPOINTS:
            callbacks.append(keras.callbacks.ModelCheckpoint(
                filepath=output_test_path + "/i_va{val_accuracy:.7f}vl{val_loss:.7f}_r" + "{:02d}".format(run_idx + 1) + "e{epoch:04d}_" + model_name + r["effect_print"] + ".h5",
                save_weights_only=False,
                monitor='val_loss',
                mode='auto',  # 'max',
                save_best_only=True))
        if SAVE_BEST_ACC_CHECKPOINTS:
            callbacks.append(keras.callbacks.ModelCheckpoint(
                filepath=output_test_path + "/i_va{val_accuracy:.7f}vl{val_loss:.7f}_r" + "{:02d}".format(run_idx + 1) + "e{epoch:04d}_" + model_name + r["effect_print"] + ".h5",
                save_weights_only=False,
                monitor='val_accuracy',
                mode='auto',
                save_best_only=True))

    # datagen.fit(x_train)
    # datagen_flow = datagen.flow(x_train, y_train, batch_size=32)

    # model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds)
    history = model.fit(
        # datagen_flow,
        # datagen.flow(x_train, y_train, batch_size=32),
        # x=x_train[:-1], #  list(np.moveaxis(x_train, -1, 0)),
        # y=y_train[:-1],
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        # validation_data=(list(np.moveaxis(x_val, -1, 0)), y_val),
        validation_data=(x_val, y_val),
        # verbose=1,
        shuffle=True,
        callbacks=callbacks,
        # validation_split=0.2,
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

    is_model_best_simple_accuracy = False
    if (r["effect_print"] not in r["test_results"]) or (current_test_best_simple_accuracy > r["test_results"][r["effect_print"]]["accuracy_simple"]):
        is_model_best_simple_accuracy = True

    test_results = r["test_results"]
    if r["effect_print"] in r["test_results"]:
        test_results[r["effect_print"]] = {
            "max_val_accuracy": max(r["test_results"][r["effect_print"]]["max_val_accuracy"], max_val_accuracy),
            "min_val_loss": min(r["test_results"][r["effect_print"]]["min_val_loss"], min_val_loss),
            "max_combined": max(r["test_results"][r["effect_print"]]["max_combined"], max_combined),
            "accuracy_simple": max(r["test_results"][r["effect_print"]]["accuracy_simple"], current_test_best_simple_accuracy)}
    else:
        test_results[r["effect_print"]] = {"max_val_accuracy": max_val_accuracy, "min_val_loss": min_val_loss, "max_combined": max_combined, "accuracy_simple": current_test_best_simple_accuracy}
    r["test_results"] = test_results

    if SAVE_MODELS and (current_test_best_simple_accuracy >= SAVE_MODELS_MIN_TEST_ACCURACY):
        # '_' + str(face_width) + 'x' + str(face_height) + 'x' + str(face_channels) + '_' + time_stamp +
        # model.save_weights('output/tests/result_mask_model_' + 'effect_' + effect_print + '_' + MODEL_NAME + '_' + str(face_width) + '_' + str(face_height) + '_' + str(face_channels) + '__' + time_stamp + '.h5')
        try:
            model_path = output_test_path + "/ta{0:03d}".format(
                int(round(current_test_best_simple_accuracy * 1000, 0))) + "tl{0:03d}".format(
                int(round(current_test_best_simple_loss * 1000, 0))) + "_i_va{:.7f}".format(
                current_val_accuracy) + "vl{:.7f}".format(current_val_loss) + "_r{0:02d}".format(
                run_idx + 1) + "e{0:04d}".format(epochs) + '_' + model_name + r["effect_print"] + '.h5'
            model.save(model_path)
            if is_model_best_simple_accuracy:
                r["model_best_simple_accuracy_path"] = model_path
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

    # results = model.evaluate(list(np.moveaxis(x_test, -1, 0)), y_test, batch_size=1)
    # results = model.evaluate(list(np.moveaxis(x_val, -1, 0)), y_val, batch_size=1)

    # results = model.evaluate(x_val, y_val, batch_size=1)
    # print("test loss, test acc:", results)

    # predictions = model.predict(list(np.moveaxis(x_test, -1, 0)))
    # predictions = model.predict(list(np.moveaxis(x_val, -1, 0)))
    # predictions = model.predict(x_train[-1].reshape((1, ) + x_train[-1].shape))
    predictions = model.predict(x_test_simple)
    print("predictions shape:", predictions.shape)
    # predictions
    # print("prediction [0, 0]:", predictions[0, 0], " should be:", y_test[0, 0, 0])
    print("prediction [0, 0]:", predictions[0], " should be:", y_test_simple[0])

    del model
    model = None
    del history
    history = None
    gc.collect()

    intermediate_models_filenames_list = glob.glob1(output_test_path, "i_*")
    for intermediate_filename in intermediate_models_filenames_list:
        model_path = os.path.join(output_test_path, intermediate_filename)
        model_intermediate = keras.models.load_model(model_path)
        results_super_intermediate = model_intermediate.evaluate(x_test_simple, y_test_simple, batch_size=1)
        print("results model_best_val_accuracy super test simple loss, test simple acc:", results_super_intermediate)
        # os.path.basename(
        rename_intermediate_filename = "ta{0:03d}".format(
            int(round(results_super_intermediate[1] * 1000, 0))) + "tl{0:03d}".format(
            int(round(results_super_intermediate[0] * 1000, 0))) + "_" + intermediate_filename
        model_path_new = os.path.join(output_test_path, rename_intermediate_filename)
        isbestacc = False
        if results_super_intermediate[1] > current_test_best_simple_accuracy:
            current_test_best_simple_accuracy = results_super_intermediate[1]
            isbestacc = True
        if results_super_intermediate[1] >= SAVE_MODELS_MIN_TEST_ACCURACY:
            try:
                os.rename(model_path, model_path_new)
                if isbestacc:
                    # model_best_simple_accuracy = model_intermediate
                    r["model_best_simple_accuracy_path"] = model_path_new
            except:
                os.remove(model_path)
        else:
            os.remove(model_path)
        del model_intermediate
        model_intermediate = None
        gc.collect()

    if current_test_best_simple_accuracy > r["test_results"][r["effect_print"]]["accuracy_simple"]:
        test_results = r["test_results"]
        test_results[r["effect_print"]]["accuracy_simple"] = current_test_best_simple_accuracy
        r["test_results"] = test_results

    # input("Press Enter to continue...")

    # img = keras.preprocessing.image.load_img("PetImages/Cat/6779.jpg", target_size=image_size)
    # img_array = keras.preprocessing.image.img_to_array(img)
    # img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    # predictions = model.predict(img_array)
    # score = predictions[0]
    # print("This image is %.2f percent cat and %.2f percent dog." % (100 * (1 - score), 100 * score))

    printTestResults(min_val_loss, max_val_accuracy, current_test_best_simple_accuracy, r)

    # return_dict["model_best_simple_accuracy_path"] = model_best_simple_accuracy_path





if __name__ == '__main__':

    # x_out = None
    # y_out = None
    # last_effect_print_right = None
    # x_out_test_simple, y_out_test_simple = None, None

    freeze_support()
    # OUTPUT_TEST_PATH = pathToScriptFolder + "/output/tests/" + datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    output_test_path = OUTPUT_TEST_PATH + datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    if not os.path.exists(output_test_path):
        os.makedirs(output_test_path)

    manager = mp.Manager()
    r = manager.dict()

    r["test_results"] = {}

    r["super_max_val_accuracy"] = 0
    r["super_max_val_accuracy_key"] = ""
    r["super_min_val_loss"] = 1
    r["super_min_val_loss_key"] = ""

    r["super_max_combined"] = 0
    r["super_max_combined_key"] = ""

    r["super_accuracy_simple"] = 0
    r["super_accuracy_simple_key"] = ""

    r["effect_print"] = ""

    r["parsed_data_last_path_list"] = []

    r["model_best_simple_accuracy_path"] = None

    for test_task in TEST_TASKS_LIST_RANDOM:
    # for test_task in TEST_TASKS_LIST_SIMPLE:
    # for test_task in TEST_TASKS_LIST:
        # for effect, value in test_task["effects"]:
        #     runTest(effect, value)
        taskTest(test_task, output_test_path, r)
        '''p = Process(target=runTest, args=(test_task, OUTPUT_TEST_PATH))
        p.start()
        p.join()
        p.close()
        del p'''
        gc.collect()
        if early_stopped:
            break



