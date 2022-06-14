import gc
import math
import os
import cv2
import numpy as np
import json
import time
import multiprocessing as mp
from multiprocessing.spawn import freeze_support

GLOBAL_FLIP = False
# GLOBAL_FLIP = True

EFFECT_COMPRESS_CUBIC = "comp"  # don't use
EFFECT_DECOMPRESS_CUBIC = "decomp"  # don't use

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
EFFECT_FLIP = "f"

EFFECT = EFFECT_GRAY
EFFECT_VALUE = 1

EXTEND_COCO = True
# EXTEND_COCO = False

#IMAGES_FOLDER_SRC = 'D:/_Projects/InfraredCamera/ImageTests/sources'
#IMAGES_FOLDER_SRC = '/tests/train_models/yolact_train/facemasknoses/train/images'
#IMAGES_FOLDER_DST = ''
#IMAGES_FOLDER_DST = IMAGES_FOLDER_SRC
#IMAGES_FOLDER_DST = IMAGES_FOLDER_SRC + '/../test1'
#IMAGES_FOLDER_DST = IMAGES_FOLDER_SRC + '/../outputs/' + EFFECT + str(EFFECT_VALUE)
#IMAGES_FOLDER_DST = IMAGES_FOLDER_SRC + '/' + EFFECT + str(EFFECT_VALUE)

PREP_SEQUENCES_LIST = [
    # [{"effect": EFFECT_GRAY, "value": 3}],
    # [{"effect": EFFECT_GRAY, "value": 0}],
    # [{"effect": EFFECT_HSV, "value": None}],
    # [{"effect": EFFECT_YCC, "value": None}],
    # [{"effect": EFFECT_LAB, "value": 3}],
    # [{"effect": EFFECT_LAB, "value": 0}],
    # [{"effect": EFFECT_BLUR, "value": 3}],
    # [{"effect": EFFECT_BLUR, "value": 1}],
    # [{"effect": EFFECT_SHARPEN, "value": 9}],
    # [{"effect": EFFECT_BRIGHTEN, "value": 10}],
    # [{"effect": EFFECT_DARKEN, "value": 10}],
    # [{"effect": EFFECT_CONTRAST_INC, "value": 10}],
    # [{"effect": EFFECT_CONTRAST_DEC, "value": 10}],
    # [{"effect": EFFECT_CONTRAST_DEC, "value": 6}],
    # [{"effect": EFFECT_SATURATE, "value": 10}],
    # [{"effect": EFFECT_DESATURATE, "value": 10}],
    # [{"effect": EFFECT_ADAPTIVE, "value": 3}],
    # [{"effect": EFFECT_NORMALIZE, "value": None}],
    # [{"effect": EFFECT_DEVIATION, "value": None}],
    # [{"effect": EFFECT_DEVIATION2, "value": None}],
    # [{"effect": EFFECT_TEST1, "value": None}],
    # [{"effect": EFFECT_TEST2, "value": None}],
    # [{"effect": EFFECT_GRAY, "value": 3}, {"effect": EFFECT_BLUR, "value": 9}],
    # [{"effect": EFFECT_GRAY, "value": 3}, {"effect": EFFECT_BLUR, "value": 1}],
    # [{"effect": EFFECT_GRAY, "value": 3}, {"effect": EFFECT_BLUR, "value": 3}],
    # [{"effect": EFFECT_GRAY, "value": 3}, {"effect": EFFECT_SHARPEN, "value": 9}],
    # [{"effect": EFFECT_GRAY, "value": 3}, {"effect": EFFECT_BRIGHTEN, "value": 10}],
    # [{"effect": EFFECT_GRAY, "value": 3}, {"effect": EFFECT_DARKEN, "value": 10}],
    # [{"effect": EFFECT_GRAY, "value": 3}, {"effect": EFFECT_CONTRAST_INC, "value": 10}],
    # [{"effect": EFFECT_GRAY, "value": 3}, {"effect": EFFECT_CONTRAST_DEC, "value": 10}],
    # [{"effect": EFFECT_GRAY, "value": 3}, {"effect": EFFECT_SATURATE, "value": 10}],
    # [{"effect": EFFECT_GRAY, "value": 3}, {"effect": EFFECT_DESATURATE, "value": 10}],
    # [{"effect": EFFECT_GRAY, "value": 3}, {"effect": EFFECT_ADAPTIVE, "value": 9}],
    # [{"effect": EFFECT_GRAY, "value": 3}, {"effect": EFFECT_ADAPTIVE, "value": 3}],
    # [{"effect": EFFECT_GRAY, "value": 3}, {"effect": EFFECT_NORMALIZE, "value": None}],
    # [{"effect": EFFECT_GRAY, "value": 3}, {"effect": EFFECT_DEVIATION, "value": None}],
    # [{"effect": EFFECT_GRAY, "value": 3}, {"effect": EFFECT_DEVIATION2, "value": None}],
]

TEST_FLIP_PREP = [
    [(EFFECT_NONE, "")],
    [(EFFECT_FLIP, "")],
    [(EFFECT_GRAY, 3)],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, "")],
]

EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_FLIP_LIST = [
    [(EFFECT_NONE, "")],
    [(EFFECT_FLIP, "")],
    [(EFFECT_TEST1, "")],
    [(EFFECT_TEST1, ""), (EFFECT_FLIP, "")],
    [(EFFECT_GRAY, 3)],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, "")],
    [(EFFECT_BLUR, 3)],
    [(EFFECT_BLUR, 3), (EFFECT_FLIP, "")],
    [(EFFECT_SHARPEN, 9)],
    [(EFFECT_SHARPEN, 9), (EFFECT_FLIP, "")],
    [(EFFECT_BRIGHTEN, 10)],
    [(EFFECT_BRIGHTEN, 10), (EFFECT_FLIP, "")],
    [(EFFECT_DARKEN, 10)],
    [(EFFECT_DARKEN, 10), (EFFECT_FLIP, "")],
    [(EFFECT_CONTRAST_INC, 10)],
    [(EFFECT_CONTRAST_INC, 10), (EFFECT_FLIP, "")],
    [(EFFECT_CONTRAST_DEC, 10)],
    [(EFFECT_CONTRAST_DEC, 10), (EFFECT_FLIP, "")],
    [(EFFECT_NORMALIZE, "")],
    [(EFFECT_NORMALIZE, ""), (EFFECT_FLIP, "")],
    [(EFFECT_DEVIATION, "")],
    [(EFFECT_DEVIATION, ""), (EFFECT_FLIP, "")],
    [(EFFECT_DEVIATION2, "")],
    [(EFFECT_DEVIATION2, ""), (EFFECT_FLIP, "")],
]

'''EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_GRAY_FLIP_LIST = [
    [(EFFECT_NONE, 3)],
    [(EFFECT_FLIP, "")],
    [(EFFECT_TEST1, 3)],
    [(EFFECT_TEST1, 3), (EFFECT_FLIP, "")],
    [(EFFECT_GRAY, 3)],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, "")],
    [(EFFECT_GRAY, 3), (EFFECT_BLUR, 3)],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_BLUR, 3)],
    [(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9)],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_SHARPEN, 9)],
    [(EFFECT_GRAY, 3), (EFFECT_BRIGHTEN, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_BRIGHTEN, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_DARKEN, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DARKEN, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_INC, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_INC, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_DEC, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_NORMALIZE, "")],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_NORMALIZE, "")],
    [(EFFECT_GRAY, 3), (EFFECT_DEVIATION, "")],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION, "")],
    [(EFFECT_GRAY, 3), (EFFECT_DEVIATION2, "")],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION2, "")],
]'''

EXTEND_DATASET_EFFECT_SEQUENCES_GRAY_FLIP_LIST = [
    [(EFFECT_GRAY, 3)],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, "")],
    [(EFFECT_GRAY, 3), (EFFECT_BLUR, 3)],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_BLUR, 3)],
    [(EFFECT_GRAY, 3), (EFFECT_SHARPEN, 9)],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_SHARPEN, 9)],
    [(EFFECT_GRAY, 3), (EFFECT_BRIGHTEN, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_BRIGHTEN, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_DARKEN, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DARKEN, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_INC, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_INC, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_CONTRAST_DEC, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_CONTRAST_DEC, 10)],
    [(EFFECT_GRAY, 3), (EFFECT_NORMALIZE, "")],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_NORMALIZE, "")],
    [(EFFECT_GRAY, 3), (EFFECT_DEVIATION, "")],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION, "")],
    [(EFFECT_GRAY, 3), (EFFECT_DEVIATION2, "")],
    [(EFFECT_GRAY, 3), (EFFECT_FLIP, ""), (EFFECT_DEVIATION2, "")],
]

PREP_PROGRAM = [
    # {"images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/headsnew/train/images', "prep_sequences_list": PREP_SEQUENCES_LIST},
    # # {"images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/headsnew/valid/images', "prep_sequences_list": PREP_SEQUENCES_LIST},
    # {
    #     "images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb/train/images',
    #     "prep_sequences_list": [[(EFFECT_GRAY, 3)], [(EFFECT_GRAY, 3), (EFFECT_FLIP, "")]],
    #     "source_coco_path": "/../",
    #     "source_coco_json": "2021-04-03-03-57-31-470616_coco_train.json",
    # },
    # {
    #     "images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb/valid/images',
    #     "prep_sequences_list": [[(EFFECT_GRAY, 3)], [(EFFECT_GRAY, 3), (EFFECT_FLIP, "")]],
    #     "source_coco_path": "/../",
    #     "source_coco_json": "2021-04-03-03-57-31-470616_coco_valid.json",
    # },
    {
        "images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb/train/images',
        "prep_sequences_list": [[(EFFECT_GRAY, 3)]],
        "source_coco_path": "/../",
        "source_coco_json": "2021-04-05-12-00-27-875589_coco_train.json",
    },
    {
        "images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb/valid/images',
        "prep_sequences_list": [[(EFFECT_GRAY, 3)]],
        "source_coco_path": "/../",
        "source_coco_json": "2021-04-05-12-00-27-875589_coco_valid.json",
    },
    # {
    #     "images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images',
    #     "prep_sequences_list": EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_FLIP_LIST,
    #     "source_coco_path": "/../",
    #     "source_coco_json": "train_via_project_5Dec2020_13h12m_coco.json",
    # },
    # {
    #     "images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images',
    #     "prep_sequences_list": EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_FLIP_LIST,
    #     "source_coco_path": "/../",
    #     "source_coco_json": "val_via_project_23Dec2020_14h27m_coco.json",
    # },
    # {
    #     "images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images',
    #     "prep_sequences_list": EXTEND_DATASET_EFFECT_SEQUENCES_GRAY_FLIP_LIST,
    #     "source_coco_path": "/../",
    #     "source_coco_json": "train_via_project_5Dec2020_13h12m_coco.json",
    # },
    # {
    #     "images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images',
    #     "prep_sequences_list": EXTEND_DATASET_EFFECT_SEQUENCES_GRAY_FLIP_LIST,
    #     "source_coco_path": "/../",
    #     "source_coco_json": "val_via_project_23Dec2020_14h27m_coco.json",
    # },
    # {
    #     "images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images',
    #     "prep_sequences_list": EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_FLIP_LIST,
    #     "source_coco_path": "/../",
    #     "source_coco_json": "2021-01-14-09-06-21-235911_coco_train.json",
    # },
    # {
    #     "images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images',
    #     "prep_sequences_list": EXTEND_DATASET_EFFECT_SEQUENCES_WITH_ORIG_TEST1_FLIP_LIST,
    #     "source_coco_path": "/../",
    #     "source_coco_json": "2021-01-14-09-06-21-235911_coco_valid.json",
    # },
    # {
    #     "images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images',
    #     "prep_sequences_list": EXTEND_DATASET_EFFECT_SEQUENCES_GRAY_FLIP_LIST,
    #     "source_coco_path": "/../",
    #     "source_coco_json": "2021-01-14-09-06-21-235911_coco_train.json",
    # },
    # {
    #     "images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images',
    #     "prep_sequences_list": EXTEND_DATASET_EFFECT_SEQUENCES_GRAY_FLIP_LIST,
    #     "source_coco_path": "/../",
    #     "source_coco_json": "2021-01-14-09-06-21-235911_coco_valid.json",
    # },
    # {
    #     "images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/test-head2/images',
    #     "prep_sequences_list": TEST_FLIP_PREP,
    #     "source_coco_path": "/../",
    #     "source_coco_json": "via_project_29Mar2021_11h53m_coco.json",
    # },
    #{"images_folder_src": 'f:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork12falsefaceswork1nb/train/images', "prep_sequences_list": PREP_SEQUENCES_LIST},
    #{"images_folder_src": 'f:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork12falsefaceswork1nb/valid/images', "prep_sequences_list": PREP_SEQUENCES_LIST},
    # {"images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images', "prep_sequences_list": PREP_SEQUENCES_LIST},
    # {"images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images', "prep_sequences_list": PREP_SEQUENCES_LIST},
    #{"images_folder_src": r"c:\Work\InfraredCamera\ThermalView\tests\train_models\yolact_train\heads\train\images", "prep_sequences_list": PREP_SEQUENCES_LIST},
    #{"images_folder_src": r"c:\Work\InfraredCamera\ThermalView\tests\train_models\yolact_train\heads\val\images", "prep_sequences_list": PREP_SEQUENCES_LIST},
    # {"images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork12ffw1a1nbgray/train/images', "prep_sequences_list": PREP_SEQUENCES_LIST},
    # {"images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork12ffw1a1nbgray/valid/images', "prep_sequences_list": PREP_SEQUENCES_LIST},
]

def formatEffectString(effect_type, effect_value):
    effect_print = effect_type
    if effect_value != "":
        effect_print += str(effect_value)
    return effect_print

#prepImages(EFFECT, EFFECT_VALUE)
# def prepImages(images_folder_src, effect_type, effect_value, images_folder_dst):
def prepImages(images_folder_src, files_list, prep_sequence, images_folder_dst, cpu_idx, cpu_count):
    # print("\n\nApplying effect " + effect_type + " value=" + str(effect_value or "") + " and saving to folder " + images_folder_dst + "\n")
    cpu_samples_count = int(math.floor(len(files_list) / cpu_count))
    cpu_start_sample_idx = cpu_idx * cpu_samples_count
    if cpu_idx < cpu_count - 1:
        cpu_end_sample_idx = cpu_start_sample_idx + cpu_samples_count
    else:
        cpu_end_sample_idx = len(files_list)

    # for idx, file in enumerate(files_list):
    for idx in range(cpu_start_sample_idx, cpu_end_sample_idx):
        file = files_list[idx]
        if file.endswith(".jpg"):
            #file_name, file_ext = file.split('.')
            frame = cv2.imread(images_folder_src + '/' + file)
            print("Processing image " + str(idx + 1) + "/" + str(len(files_list)))
            for prep in prep_sequence:
                effect_type, effect_value = prep
                if effect_type == EFFECT_GRAY:
                    if (effect_value is None) or (effect_value >= 3):
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = np.zeros_like(frame, dtype=np.uint8)
                        frame[:, :, effect_value] = tmp
                        del tmp
                elif effect_type == EFFECT_HSV:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                elif effect_type == EFFECT_LAB:
                    if effect_value >= 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
                    else:
                        tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
                        frame = np.zeros_like(frame)
                        frame[:, :, effect_value] = tmp[:, :, 0]
                        del tmp
                elif effect_type == EFFECT_YCC:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                elif effect_type == EFFECT_COMPRESS_CUBIC:
                    frame = cv2.resize(frame, (int(frame.shape[1] * (1 - effect_value / 100)), int(frame.shape[0] * (1 - effect_value / 100))), cv2.INTER_CUBIC)
                elif effect_type == EFFECT_DECOMPRESS_CUBIC:
                    frame = cv2.resize(frame, (int(frame.shape[1] * (1 + effect_value / 100)), int(frame.shape[0] * (1 + effect_value / 100))), cv2.INTER_CUBIC)
                elif effect_type == EFFECT_BLUR:
                    frame = cv2.GaussianBlur(frame, (effect_value * 2 + 1, effect_value * 2 + 1), 0)
                elif effect_type == EFFECT_SHARPEN:
                    frame_blurred = cv2.GaussianBlur(frame, (effect_value * 2 + 1, effect_value * 2 + 1), 0)
                    frame = cv2.addWeighted(frame, 1.5, frame_blurred, -0.5, 0)
                    del frame_blurred
                elif effect_type == EFFECT_BRIGHTEN:
                    to1 = False
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        to1 = True
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) + ((255 - hsv[:, :, 2].astype(np.float32)) * (effect_value / 100)), 0, 255).astype(np.uint8)
                    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    del hsv
                    if to1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                elif effect_type == EFFECT_DARKEN:
                    to1 = False
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        to1 = True
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) - (hsv[:, :, 2].astype(np.float32) * (effect_value / 100)), 0, 255).astype(np.uint8)
                    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    del hsv
                    if to1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                elif effect_type == EFFECT_CONTRAST_INC:
                    frame = np.clip((frame.astype(np.float32) + ((127 + frame.astype(np.float32)) * (effect_value / 100))), 0, 255).astype(np.uint8)
                elif effect_type == EFFECT_CONTRAST_DEC:
                    frame = np.clip((frame.astype(np.float32) + ((127 - frame.astype(np.float32)) * (effect_value / 100))), 0, 255).astype(np.uint8)
                elif effect_type == EFFECT_SATURATE:
                    to1 = False
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        to1 = True
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    hsv[:, :, 1] = np.clip((hsv[:, :, 1].astype(np.float32) * ((100 + effect_value) / 100)), 0, 255).astype(np.uint8)
                    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    del hsv
                    if to1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                elif effect_type == EFFECT_DESATURATE:
                    to1 = False
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        to1 = True
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    hsv[:, :, 1] = np.clip((hsv[:, :, 1].astype(np.float32) * ((100 - effect_value) / 100)), 0, 255).astype(np.uint8)
                    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    del hsv
                    if to1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                elif effect_type == EFFECT_ADAPTIVE:
                    to3 = False
                    if len(frame.shape) > 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        to3 = True
                    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, effect_value, 2)
                    #frame = cv2.Laplacian(frame, cv2.CV_8U)
                    #frame = cv2.medianBlur(frame, 15)
                    if to3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif effect_type == EFFECT_NORMALIZE:
                    frame = cv2.normalize(frame, frame, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                elif effect_type == EFFECT_DEVIATION:
                    frame = frame.astype(np.float32) / 255
                    frame -= frame.mean()
                    frame /= frame.std()
                    frame = cv2.normalize(frame, frame, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    frame = (frame * 255).astype(np.uint8)
                elif effect_type == EFFECT_DEVIATION2:
                    mean = frame.mean(axis=(0, 1))
                    std = frame.std(axis=(0, 1))
                    frame = frame.astype(np.float32)
                    if len(frame.shape) > 2:
                        frame[..., 0] -= mean[0]
                        frame[..., 1] -= mean[1]
                        frame[..., 2] -= mean[2]
                    else:
                        frame[...] -= mean
                    if len(frame.shape) > 2:
                        frame[..., 0] /= std[0]
                        frame[..., 1] /= std[1]
                        frame[..., 2] /= std[2]
                    else:
                        frame[...] /= std
                    frame = cv2.normalize(frame, frame, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    frame = frame.astype(np.uint8)
                elif effect_type == EFFECT_TEST1:
                    img_B = frame[:, :, 0]
                    img_G = frame[:, :, 1]
                    img_R = frame[:, :, 2]
                    img_BG = cv2.addWeighted(img_B, 0.5, img_G, 0.5, 0)
                    img_BR = cv2.addWeighted(img_B, 0.5, img_R, 0.5, 0)
                    img_GR = cv2.addWeighted(img_G, 0.5, img_R, 0.5, 0)
                    img_BBGR = cv2.addWeighted(img_B, 0.5, img_GR, 0.5, 0)
                    img_BGGR = cv2.addWeighted(img_G, 0.5, img_BR, 0.5, 0)
                    img_BGRR = cv2.addWeighted(img_R, 0.5, img_BG, 0.5, 0)
                    # frame = img_BBGR | img_BGGR | img_BGRR
                    frame = cv2.merge((img_BBGR, img_BGGR, img_BGRR))
                    del img_B
                    del img_G
                    del img_R
                    del img_BG
                    del img_BR
                    del img_GR
                    del img_BBGR
                    del img_BGGR
                    del img_BGRR
                elif effect_type == EFFECT_TEST2:
                    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    img_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    # frame = img_gray | img_hsv[:, :, 2] | img_lab[:, :, 0]
                    frame = cv2.merge((img_gray, img_hsv[:, :, 2], img_lab[:, :, 0]))
                    del img_gray
                    del img_hsv
                    del img_lab
                elif effect_type == EFFECT_FLIP:
                    frame = cv2.flip(frame, 1)

            cv2.imwrite(images_folder_dst + '/' + file, frame)
            del frame
            gc.collect()


def prepList(prep_set):
    images_folder_src = prep_set["images_folder_src"]
    prep_sequences_list = prep_set["prep_sequences_list"]
    if "source_coco_json" in prep_set:
        coco_json = prep_set["source_coco_json"]
    else:
        coco_json = None
    if "source_coco_path" in prep_set:
        coco_path = prep_set["source_coco_path"]
    else:
        coco_path = None

    effect_folders = []
    # str_effects_folder = ""
    print("\n\n\n\nParsing folder " + images_folder_src + "\n")
    for prep_sequence in prep_sequences_list:
        effect_folder = ""
        for prep in prep_sequence:
            prep_effect, prep_value = prep
            effect_folder += formatEffectString(prep_effect, prep_value)
        images_folder_dst = images_folder_src + '/' + effect_folder
        effect_folders.append(effect_folder)
        # str_effects_folder += effect_folder
        # for prep in prep_task["prep_sequence"]:
        #     images_folder_dst = images_folder_src + '/' + prep["effect"] + (str(prep["value"]) if (prep["value"] is not None) else "")
        #     prepImages(images_folder_src, prep["effect"], prep["value"], images_folder_dst)
        if not os.path.exists(images_folder_dst):
            os.makedirs(images_folder_dst)
        files_list = os.listdir(images_folder_src)
        cpu_count = mp.cpu_count()
        cpu_count = 8
        print(f"Found {cpu_count} CPUs")
        start = time.time()
        prep_processes = []
        for cpu_idx in range(cpu_count):
            p = mp.Process(target=prepImages, args=(images_folder_src, files_list, prep_sequence, images_folder_dst, cpu_idx, cpu_count,))
            p.start()
            prep_processes += [p]
        '''for p in prep_processes:
            p.join()'''
        [p.join() for p in prep_processes]
        # map(lambda p: p.join(), prep_processes)
        '''for p in prep_processes:
            p.close()'''
        [p.close() for p in prep_processes]
        # [p.wait() for p in prep_processes]
        for p in prep_processes:
            del p
        gc.collect()

        # map(lambda p: p.close(), prep_processes)
        end = time.time()
        print(f"Runtime of general dataset prep is {end - start}")
        # prepImages(images_folder_src, files_list, prep_sequence, images_folder_dst)

    if EXTEND_COCO and (coco_path is not None) and (coco_json is not None):
        coco_input_path = images_folder_src + coco_path + coco_json
        # coco_output_path = images_folder_src + coco_output
        with open(coco_input_path) as json_file:
            coco_entry = json.load(json_file)
            new_images = []
            new_annotations = []
            for image in coco_entry["images"]:
                old_image_original_id = image["id"]
                '''don't add original images new_image = {**image}
                new_image_original_id = len(new_images) + 1
                new_image["id"] = new_image_original_id
                new_images.append(new_image)'''
                new_images_ids = []
                new_flip_images = []
                for effect_folder in effect_folders:
                    new_image = {**image}
                    new_image_id = len(new_images) + 1
                    new_image["id"] = new_image_id
                    new_images_ids.append(new_image_id)
                    if GLOBAL_FLIP or (EFFECT_FLIP in effect_folder):
                        new_flip_images.append(new_image)
                    new_image["file_name"] = effect_folder + '/' + new_image["file_name"]
                    new_images.append(new_image)
                for annotation in coco_entry["annotations"]:
                    if annotation["image_id"] == old_image_original_id:
                        '''don't add original images new_annotation = {**annotation}
                        new_annotation["image_id"] = new_image_original_id
                        new_annotation["id"] = len(new_annotations) + 1
                        new_annotations.append(new_annotation)'''
                        for new_image_id in new_images_ids:
                            new_annotation = {**annotation}
                            new_annotation["image_id"] = new_image_id
                            new_annotation["id"] = len(new_annotations) + 1
                            for new_flip_image in new_flip_images:
                                if new_image_id == new_flip_image["id"]:
                                    new_annotation["bbox"][0] = new_flip_image["width"] - new_annotation["bbox"][0] - new_annotation["bbox"][2]
                                    flipped_segmentation = []
                                    if len(new_annotation["segmentation"]) == 1:
                                        new_annotation["segmentation"] = new_annotation["segmentation"][0]
                                    for value_idx, contour_value in enumerate(new_annotation["segmentation"]):
                                        if (value_idx % 2) == 0:
                                            contour_value = new_flip_image["width"] - contour_value
                                        flipped_segmentation.append(contour_value)
                                    new_annotation["segmentation"] = [flipped_segmentation]
                            new_annotations.append(new_annotation)
            new_coco_entry = {
                "info": coco_entry["info"],
                "images": new_images,
                "annotations": new_annotations,
                "licenses": coco_entry["licenses"],
                "categories": coco_entry["categories"],
            }
            if GLOBAL_FLIP:
                effect_folders.insert(0, "flip")
            with open(images_folder_src + coco_path + "ext-" + ("".join(effect_folders)) + "_" + coco_json, 'w') as outfile:
                json.dump(new_coco_entry, outfile, indent=4)

#prepList(PREP_SEQUENCES_LIST, IMAGES_FOLDER_SRC)

def prepProgram(prog):
    for prep_set in prog:
        prepList(prep_set)

if __name__ == '__main__':

    freeze_support()

    prepProgram(PREP_PROGRAM)








# coco_train = {"info": info, "images": images_train, "annotations": annotations_train, "licenses": licenses, "categories": categories}