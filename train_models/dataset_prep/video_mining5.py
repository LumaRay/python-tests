import gc
import random
from datetime import datetime
import os
from math import sqrt

import pafy, cv2, numpy as np

import pathlib
pathToScriptFolder = str(pathlib.Path().absolute())

import json

# HEAD_SIZE_MIN = 50
# HEAD_SIZE_MIN = 30
HEAD_SIZE_MIN = 20
# HEAD_SIZE_MIN = 10

# FRAME_STEP = 30
# FRAME_STEP = 200
# FRAME_STEP = 2000
FRAME_STEP = 3000
#frame_shift = 7
# FRAMES_PROCESS_PER_VIDEO = 800
# FRAMES_PROCESS_PER_VIDEO = 8000
FRAMES_PROCESS_PER_VIDEO = 40000
# FRAMES_SAVE_PER_VIDEO = 400
# FRAMES_SAVE_PER_VIDEO = 4000
FRAMES_SAVE_PER_VIDEO = 20000
# OBJECTS_SAVE_PER_VIDEO = 400
# OBJECTS_SAVE_PER_VIDEO = 4000
OBJECTS_SAVE_PER_VIDEO = 20000

MAX_ERRORS_PER_VIDEO = 1

SAVE_PEOPLE = False

# MAX_FRAME_MEMORY_SIZE = 10000000
MAX_FRAME_MEMORY_SIZE = 100000000

#HEAD_RECTIFY_RATIO = 1
HEAD_RECTIFY_RATIO = 0.5
MIN_RATIO_AREA_ORIGINAL_TO_HULL = 0.9
# MIN_RATIO_AREA_ORIGINAL_TO_HULL = 0.7
MIN_RATIO_PERIMETER_HULL_TO_ORIGINAL = 0.9
# MIN_RATIO_PERIMETER_HULL_TO_ORIGINAL = 0.8
# MIN_RATIO_ORIGINAL_MIN_RECT_PROPORTION = 0.6
MIN_RATIO_ORIGINAL_MIN_RECT_PROPORTION = 0.57
# MIN_RATIO_ORIGINAL_MIN_RECT_PROPORTION = 0.53
# MIN_RATIO_ORIGINAL_MIN_RECT_PROPORTION = 0.5
# MIN_RATIO_ORIGINAL_MIN_RECT_PROPORTION = 0.48
# MIN_RATIO_ORIGINAL_MIN_RECT_PROPORTION = 0.3
# MAX_TOP_NONZEROS_RATIO = 0.3
MAX_TOP_NONZEROS_RATIO = 0.4
# MAX_TOP_NONZEROS_RATIO = 0.5
INTERSECT_WITH_ELLIPSE = True

people_yolact_trained_model = '../../../face_detection/yolact/weights/yolact_resnet50_54_800000.pth'
#heads_yolact_trained_model = '../../../face_detection/yolact/weights/yolact_heads_3399_34000.pth'
heads_detector_heads_yolact_trained_model = '../../../face_detection/yolact/weights/yolact_heads_3399_34000.pth'
# heads_detector_maskfaces_yolact_trained_model = '../../../face_detection/yolact/weights/yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbgray_17_38000.pth'
heads_detector_maskfaces_yolact_trained_model = '../../../face_detection/yolact/weights/yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbgray640_64_146000.pth'
heads_detector_facemasknoses_yolact_trained_model = '../../../face_detection/yolact/weights/yolact_facemasknosesgray_805_54000.pth'

HEADS_DETECTOR_HEADS = "heads512"
HEADS_DETECTOR_MASKFACES = "maskfaces"
HEADS_DETECTOR_FACEMASKNOSES = "facemasknoses"

HEAD_DETECTOR = HEADS_DETECTOR_MASKFACES

HEADS_DETECTOR_PRECONVERT_GRAY = True

HEAD_USE_HEIGHT_CORRECTION = False  # head_h = max(head_h, head_w * 4 / 3)
HEAD_USE_HEIGHT_CORRECTION2 = True

SORT_BY_MASK_DETECTION = True
# SORT_BY_MASK_DETECTION = False
SORT_BY_MASK_DETECTION_MULTICLASS = True
SORT_BY_MASK_DETECTION_MULTICLASS_EXTRA = True
# MULTICLASS_EXTRA_MINIMUM_ACCURACY = 0
MULTICLASS_EXTRA_MINIMUM_ACCURACY = 0.1
# MULTICLASS_EXTRA_MINIMUM_ACCURACY = 0.5
# MULTICLASS_EXTRA_MINIMUM_ACCURACY = 0.9

if HEAD_DETECTOR == HEADS_DETECTOR_HEADS:
    heads_yolact_trained_model = heads_detector_heads_yolact_trained_model
elif HEAD_DETECTOR == HEADS_DETECTOR_MASKFACES:
    heads_yolact_trained_model = heads_detector_maskfaces_yolact_trained_model
elif HEAD_DETECTOR == HEADS_DETECTOR_FACEMASKNOSES:
    heads_yolact_trained_model = heads_detector_facemasknoses_yolact_trained_model

yolact_top_k = 150
heads_yolact_score_threshold = 0.0
people_yolact_score_threshold = 0.15

# https://github.com/dbolya/yolact
from face_detection.yolact.yolact_heads import Yolact as heads_Yolact
from face_detection.yolact.yolact_people import Yolact as people_Yolact
import torch
import torch.backends.cudnn as cudnn
from face_detection.yolact.utils.functions import SavePath as yolact_SavePath
#from face_detection.yolact.config import COLORS as yolact_COLORS
from face_detection.yolact.config_heads import cfg_heads as heads_yolact_cfg, set_cfg_heads as heads_yolact_set_cfg, COLORS as yolact_COLORS
from face_detection.yolact.config_people import cfg_people as people_yolact_cfg, set_cfg_people as people_yolact_set_cfg
from face_detection.yolact.utils.augmentations import FastBaseTransform as yolact_FastBaseTransform
# from face_detection.yolact.utils.functions import MovingAverage
#from face_detection.yolact.layers.output_utils import undo_image_transformation as yolact_undo_image_transformation
from face_detection.yolact.layers.output_utils_heads import postprocess as heads_yolact_postprocess, undo_image_transformation as yolact_undo_image_transformation
from face_detection.yolact.layers.output_utils_people import postprocess as people_yolact_postprocess
from collections import defaultdict

yolact_color_cache = defaultdict(lambda: {})
#yolact_trained_model = '../face_detection/yolact/weights/yolact_base_54_800000.pth'
#yolact_trained_model = 'f:\\Work\\InfraredCamera\\ThermalView\\face_detection\\yolact\\weights\\yolact_resnet50_54_800000.pth'
#yolact_trained_model = pathToScriptFolder + '/face_detection/yolact/weights/yolact_maskfaces_5117_51173_interrupt.pth'
#yolact_trained_model = pathToScriptFolder + '/face_detection/yolact/weights/yolact_maskfaces_10952_109522_interrupt.pth'
#yolact_trained_model = pathToScriptFolder + '/face_detection/yolact/weights/yolact_darknet53_54_800000.pth'
yolact_cuda = True
yolact_fast_nms = True
yolact_cross_class_nms = False
yolact_display_masks = True
yolact_display_bboxes = True
yolact_display_text = True
yolact_display_scores = True
yolact_display_lincomb = False
yolact_mask_proto_debug = False
yolact_video_multiframe = 1
yolact_crop = True
# yolact_crop = False

heads_yolact_model_path = yolact_SavePath.from_str(heads_yolact_trained_model)
heads_yolact_config = heads_yolact_model_path.model_name + '_config'
heads_yolact_set_cfg(heads_yolact_config)

people_yolact_model_path = yolact_SavePath.from_str(people_yolact_trained_model)
people_yolact_config = people_yolact_model_path.model_name + '_config'
people_yolact_set_cfg(people_yolact_config)

if SORT_BY_MASK_DETECTION:
    if SORT_BY_MASK_DETECTION_MULTICLASS:
        import tensorflow as tf
        from tensorflow import keras
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
        CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']
        MULTICLASS_EXTRA_FACE_WIDTH = 128
        MULTICLASS_EXTRA_FACE_HEIGHT = 128
        multiclass_extra_mask_model = keras.models.load_model('../../../face_detection/uem_mask/ta922tl168_i_va0.999vl0.002_r21e75_1-20-0-128l256l512l728l1024-128x128x3--g3--bl3--sh9--br10--dr10--cn10--dc10--nr--dv--vv.h5')

with torch.no_grad():
    if yolact_cuda:
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    heads_yolact_net = heads_Yolact()
    heads_yolact_net.load_weights(heads_yolact_trained_model)
    heads_yolact_net.eval()

    if yolact_cuda:
        heads_yolact_net = heads_yolact_net.cuda()

    people_yolact_net = people_Yolact()
    people_yolact_net.load_weights(people_yolact_trained_model)
    people_yolact_net.eval()

    if yolact_cuda:
        people_yolact_net = people_yolact_net.cuda()

    # evaluate(net, dataset)
    # def evaluate(net:Yolact, dataset, train_mode=False):
    people_yolact_net.detect.use_fast_nms = yolact_fast_nms
    people_yolact_net.detect.use_cross_class_nms = yolact_cross_class_nms
    heads_yolact_net.detect.use_fast_nms = yolact_fast_nms
    heads_yolact_net.detect.use_cross_class_nms = yolact_cross_class_nms
    people_yolact_cfg.mask_proto_debug = yolact_mask_proto_debug
    heads_yolact_cfg.mask_proto_debug = yolact_mask_proto_debug

    # evalvideo(net, args.video)
    # def evalvideo(net:Yolact, path:str, out_path:str=None):

    class yolact_CustomDataParallel(torch.nn.DataParallel):
        """ A Custom Data Parallel class that properly gathers lists of dictionaries. """

        def gather(self, outputs, output_device):
            # Note that I don't actually want to convert everything to the output_device
            return sum(outputs, [])

    heads_yolact_net = yolact_CustomDataParallel(heads_yolact_net).cuda()
    people_yolact_net = yolact_CustomDataParallel(people_yolact_net).cuda()
    yolact_transform = torch.nn.DataParallel(yolact_FastBaseTransform()).cuda()
    yolact_extract_frame = lambda x, i: (x[0][i] if x[1][i]['detection'] is None else x[0][i].to(x[1][i]['detection']['box'].device), [x[1][i]])


#IMAGES_FOLDER = '../tests/work/source/'
#FRAMES_FOLDER = 'frames'
#OBJECTS_FOLDER = 'objects'

#IMAGES_FOLDER = 'F:/Work/InfraredCamera/ThermalView/tests/work/false_faces_work1/'
##PARSED_FOLDER = 'false_faces_work1/heads'

##heads_yolact_trained_model = '../face_detection/yolact/weights/yolact_heads_3399_34000.pth'

#PARSED_FOLDER = 'false_faces_work1/maskfacesnewwork0312added1nb'
#heads_yolact_trained_model = '../face_detection/yolact/weights/yolact_maskfacesnewwork0312added1nb_302_192162_interrupt.pth'

# USE_YT = False
# USE_VIDEO_LIST = False
# READ_VIDEO_LIST_FROM_FILE = False
# USE_IMAGES_FOLDER = True
# USE_PERSON_TRACK_FILTERING = True

F_IMAGES_FOLDER = "images_folder"
F_PERSON_TRACK_FILTERING = "person_track_filtering"
F_PARSED_FOLDER_NAME = "parsed_folder_name"
F_TASK_TYPE_LIST_FROM_FILE = "list_from_file"
F_LIST_FILE_PATH = "list_file_path"
F_SOURCE_PATH = "source_path"
F_SOURCES_LIST = "sources_list"

F_TASK_TYPE = "task_type"
TASK_TYPE_YOUTUBE = "youtube"
TASK_TYPE_YOUTUBE_LIST = "youtube_list"
TASK_TYPE_VIDEO = "video"
TASK_TYPE_VIDEO_LIST = "video_list"
TASK_TYPE_IMAGES_FOLDER = "images_folder"

task_list = [
    {F_TASK_TYPE: TASK_TYPE_YOUTUBE_LIST, F_PERSON_TRACK_FILTERING: True, F_PARSED_FOLDER_NAME: "v2_yt_sourcestest6", F_SOURCES_LIST: ['https://www.youtube.com/watch?v=QjjBNU88jtk']},
    # {F_TASK_TYPE: TASK_TYPE_IMAGES_FOLDER, F_PERSON_TRACK_FILTERING: False, F_IMAGES_FOLDER: "/../raw_images/vk_sources_test1/", F_PARSED_FOLDER_NAME: "v2_vk_sources_test1"},
    # {F_TASK_TYPE: TASK_TYPE_IMAGES_FOLDER, F_PERSON_TRACK_FILTERING: True, F_IMAGES_FOLDER: "/../raw_images/2021_04_14/full/color/source/", F_PARSED_FOLDER_NAME: "v2_exhib_source_2021_04_14"},
    # {F_TASK_TYPE: TASK_TYPE_IMAGES_FOLDER, F_PERSON_TRACK_FILTERING: True, F_IMAGES_FOLDER: "/../raw_images/work/source_2020_11_17/", F_PARSED_FOLDER_NAME: "v2_source_2020_11_17"},
    # {F_TASK_TYPE: TASK_TYPE_IMAGES_FOLDER, F_PERSON_TRACK_FILTERING: True, F_IMAGES_FOLDER: "/../raw_images/work/source_2020_11_19/", F_PARSED_FOLDER_NAME: "v2_source_2020_11_19"},
    # {F_TASK_TYPE: TASK_TYPE_IMAGES_FOLDER, F_PERSON_TRACK_FILTERING: True, F_IMAGES_FOLDER: "/../raw_images/work/source_2020_11_25_200ms/", F_PARSED_FOLDER_NAME: "v2_source_2020_11_25_200ms"},
    # {F_TASK_TYPE: TASK_TYPE_IMAGES_FOLDER, F_PERSON_TRACK_FILTERING: True, F_IMAGES_FOLDER: "/../raw_images/work/source_2020_11_25_300ms/", F_PARSED_FOLDER_NAME: "v2_source_2020_11_25_300ms"},
    # {F_TASK_TYPE: TASK_TYPE_IMAGES_FOLDER, F_PERSON_TRACK_FILTERING: True, F_IMAGES_FOLDER: "/../raw_images/work/source_2020_11_25_1000ms/", F_PARSED_FOLDER_NAME: "v2_source_2020_11_25_1000ms"},
    # {F_TASK_TYPE: TASK_TYPE_IMAGES_FOLDER, F_PERSON_TRACK_FILTERING: True, F_IMAGES_FOLDER: "/../raw_images/work/source_2020_11_25_1000ms5max/", F_PARSED_FOLDER_NAME: "v2_source_2020_11_25_1000ms5max"},
    # {F_TASK_TYPE: TASK_TYPE_IMAGES_FOLDER, F_PERSON_TRACK_FILTERING: True, F_IMAGES_FOLDER: "/../raw_images/work/source_2020_11_25_3000ms/", F_PARSED_FOLDER_NAME: "v2_source_2020_11_25_3000ms"},
    # {F_TASK_TYPE: TASK_TYPE_IMAGES_FOLDER, F_PERSON_TRACK_FILTERING: True, F_IMAGES_FOLDER: "/../raw_images/work/source_2020_11_25_3000ms5max/", F_PARSED_FOLDER_NAME: "v2_source_2020_11_25_3000ms5max"},
    # {F_TASK_TYPE: TASK_TYPE_IMAGES_FOLDER, F_PERSON_TRACK_FILTERING: True, F_IMAGES_FOLDER: "/../raw_images/work/source_2020_11_25_noms/", F_PARSED_FOLDER_NAME: "v2_source_2020_11_25_noms"},
    # {F_TASK_TYPE: TASK_TYPE_IMAGES_FOLDER, F_PERSON_TRACK_FILTERING: False, F_IMAGES_FOLDER: "/../raw_images/vk_sources1/", F_PARSED_FOLDER_NAME: "v2_vk_sources1"},
    # {F_TASK_TYPE: TASK_TYPE_IMAGES_FOLDER, F_PERSON_TRACK_FILTERING: False, F_IMAGES_FOLDER: "/../raw_images/vk_sources2/", F_PARSED_FOLDER_NAME: "v2_vk_sources2"},
    # {F_TASK_TYPE: TASK_TYPE_YOUTUBE_LIST, F_PERSON_TRACK_FILTERING: True, F_LIST_FILE_PATH: "/../yt_video_sources1-6.lst", F_PARSED_FOLDER_NAME: "v2_yt_sources1to6", F_TASK_TYPE_LIST_FROM_FILE: True},
    # {F_TASK_TYPE: TASK_TYPE_YOUTUBE, F_PERSON_TRACK_FILTERING: True, F_PARSED_FOLDER_NAME: "v2_yt_source1", F_SOURCE_PATH: 'https://www.youtube.com/watch?v=RsrpV-GweKc'},
    # {F_TASK_TYPE: TASK_TYPE_YOUTUBE_LIST, F_PERSON_TRACK_FILTERING: True, F_PARSED_FOLDER_NAME: "v2_yt_sourcestest5", F_SOURCES_LIST: [
    #     'https://www.youtube.com/watch?v=xn7wPPSh6yI',
    #     'https://www.youtube.com/watch?v=FmI6pcE5GPU',
    #     'https://www.youtube.com/watch?v=SxIUyECUEik',
    #     'https://www.youtube.com/watch?v=3LwWl2wU4tQ',
    #     'https://www.youtube.com/watch?v=uSLZfNteDxM',
    # ]},
    # {F_TASK_TYPE: TASK_TYPE_VIDEO, F_PERSON_TRACK_FILTERING: True, F_PARSED_FOLDER_NAME: "v2_vid_source1", F_SOURCE_PATH: 'D:\\_yt_data\\720p_6g9dSth3p_w.mp4'},
    # {F_TASK_TYPE: TASK_TYPE_VIDEO_LIST, F_PERSON_TRACK_FILTERING: True, F_PARSED_FOLDER_NAME: "v2_vid_sourcestest9", F_SOURCES_LIST: [
    #     'D:\\_yt_data\\720p_6g9dSth3p_w.mp4',
    #     'D:\\_yt_data\\720p_6vuFh6NNa70.mp4',
    #     'D:\\_yt_data\\720p_33XDY0cb86Q.mp4',
    #     'D:\\_yt_data\\720p_BkBumUzv73U.mp4',
    #     'D:\\_yt_data\\720p_FmI6pcE5GPU.mp4',
    #     'D:\\_yt_data\\720p_KNM8w4kFiB0.mp4',
    #     'D:\\_yt_data\\720p_MJ6uRDa1oJ0.mp4',
    #     'D:\\_yt_data\\720p_T96LyBlSRM4.mp4',
    #     'D:\\_yt_data\\720p_vU0-R1w-hcQ.mp4',
    # ]},
]

# IMAGES_FOLDER = pathToScriptFolder + '/../raw_images/work/'
# PARSED_FOLDER_NAME = 'v2_maskfacesnewwork12ffw1a1nbncv2gray_using_' + HEAD_DETECTOR

# IMAGES_FOLDER = pathToScriptFolder + '/../test_set/'
# PARSED_FOLDER_NAME = 'worktestset_using_' + HEAD_DETECTOR



for task in task_list:

    task_type = task[F_TASK_TYPE]
    parsed_folder_name = task[F_PARSED_FOLDER_NAME]

    if task.get(F_IMAGES_FOLDER) is not None:
        images_folder = pathToScriptFolder + task[F_IMAGES_FOLDER]
    else:
        images_folder = ""
    if task.get(F_TASK_TYPE_LIST_FROM_FILE) is not None:
        list_from_file = task[F_TASK_TYPE_LIST_FROM_FILE]
    else:
        list_from_file = False
    if task.get(F_LIST_FILE_PATH) is not None:
        list_file_path = pathToScriptFolder + task[F_LIST_FILE_PATH]
    else:
        list_file_path = ""
    if task.get(F_PERSON_TRACK_FILTERING) is not None:
        person_track_filtering = task[F_PERSON_TRACK_FILTERING]
    else:
        person_track_filtering = False
    if task.get(F_SOURCE_PATH) is not None:
        source_path = task[F_SOURCE_PATH]
    else:
        source_path = ""
    if task.get(F_SOURCES_LIST) is not None:
        sources_list = task[F_SOURCES_LIST]
    else:
        sources_list = []

    FRAMES_FOLDER = pathToScriptFolder + '/../parsed_data/' + parsed_folder_name + '/frames/'
    OBJECTS_FOLDER = pathToScriptFolder + '/../parsed_data/' + parsed_folder_name + '/objects/'

    if not os.path.exists(FRAMES_FOLDER):
        os.makedirs(FRAMES_FOLDER)

    if not os.path.exists(OBJECTS_FOLDER):
        os.makedirs(OBJECTS_FOLDER)

    if SORT_BY_MASK_DETECTION:
        if SORT_BY_MASK_DETECTION_MULTICLASS:
            for cls in CLASS_NAMES:
                if not os.path.exists(OBJECTS_FOLDER + cls + "/"):
                    os.makedirs(OBJECTS_FOLDER + cls + "/")
        else:
            if not os.path.exists(OBJECTS_FOLDER + "mask/"):
                os.makedirs(OBJECTS_FOLDER + "mask/")
            if not os.path.exists(OBJECTS_FOLDER + "nomask/"):
                os.makedirs(OBJECTS_FOLDER + "nomask/")

    video_source_list = []
    if task_type == TASK_TYPE_YOUTUBE:
        #url = 'https://www.youtube.com/watch?v=xn7wPPSh6yI'
        video_source = source_path
        #url = 'https://youtu.be/SxIUyECUEik'
        #url = 'https://youtu.be/3LwWl2wU4tQ'
        #url = 'https://youtu.be/uSLZfNteDxM'
    elif task_type == TASK_TYPE_YOUTUBE_LIST:
        if list_from_file:
            video_source_list_file = open(list_file_path, "r")
            video_source_list = video_source_list_file.read().splitlines()
            random.shuffle(video_source_list)
        else:
            video_source_list = sources_list
    elif task_type == TASK_TYPE_VIDEO:
        #video_file = 'D:\\_yt_data\\33XDY0cb86Q.mp4'
        video_source = source_path
    elif task_type == TASK_TYPE_VIDEO_LIST:
        video_source_list = sources_list
    else:
        pass

    images_list = []
    if (task_type == TASK_TYPE_IMAGES_FOLDER) and (images_folder != ""):
        # for file in os.listdir(IMAGES_FOLDER):
        #     if file.endswith(".jpg"):
        #         images_list.append(IMAGES_FOLDER + file)
        for root, subdirs, files in os.walk(images_folder):
            for file in files:
                if file.endswith(".jpg"):
                    images_list.append(root + '/' + file)

    i = 0
    processed_frames_count = 0
    saved_frames_count = 0
    saved_objects_count = 0
    cap = None
    video_source_idx = 0
    errors_per_video = 0
    last_heads_bboxes = []

    frame_shift = random.randint(0, FRAME_STEP - 1)

    while (True and ((processed_frames_count < len(images_list)) or (task_type != TASK_TYPE_IMAGES_FOLDER))):
        # from numba import cuda
        # device = cuda.get_current_device()
        # cuda.close() or device.reset() This leaves tensorflow in a bad state
        if ((task_type == TASK_TYPE_VIDEO_LIST) or (task_type == TASK_TYPE_YOUTUBE_LIST)) and ((errors_per_video >= MAX_ERRORS_PER_VIDEO) or (saved_frames_count >= FRAMES_SAVE_PER_VIDEO) or (saved_objects_count >= OBJECTS_SAVE_PER_VIDEO) or (processed_frames_count >= FRAMES_PROCESS_PER_VIDEO)):
            errors_per_video = 0
            video_source_idx += 1
            processed_frames_count = 0
            saved_frames_count = 0
            saved_objects_count = 0
            i = 0
            if (cap is not None) and cap.isOpened():
                cap.release()
            cap = None
            continue

        if (task_type == TASK_TYPE_IMAGES_FOLDER) and (images_folder != ""):
            print("Parsing image " + str(processed_frames_count) + " of " + str(len(images_list)) + ": " + images_list[processed_frames_count])
            cap = cv2.VideoCapture(images_list[processed_frames_count])

        if cap is None:
            if video_source_idx >= len(video_source_list):
                break
            if (task_type == TASK_TYPE_VIDEO_LIST) or (task_type == TASK_TYPE_YOUTUBE_LIST):
                video_source = video_source_list[video_source_idx]
            try:
                if (task_type == TASK_TYPE_YOUTUBE) or (task_type == TASK_TYPE_YOUTUBE_LIST):
                    try:
                        vPafy = pafy.new(video_source)
                    except Exception as e:
                        print(e)
                        video_source_idx += 1
                        processed_frames_count = 0
                        saved_frames_count = 0
                        saved_objects_count = 0
                        errors_per_video = 0
                        i = 0
                        if cap.isOpened():
                            cap.release()
                        cap = None
                        continue
                    # play = vPafy.getbest()
                    # play = vPafy.getbest(preftype='video:mp4@1920x1080')
                    # play = vPafy.getbestvideo()
                    # cap = cv2.VideoCapture(play.url)
                    for vstream in vPafy.videostreams:
                        if vstream.dimensions == (1920, 1080):
                        # if vstream.dimensions == (3840, 2160):
                        # if vstream.dimensions == (2560, 1440):
                            cap = cv2.VideoCapture(vstream.url)
                    #         # break
                else:
                    cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
            except:
                cap = None

        if cap is None:
            continue

        ret, frame = cap.read()

        if not ret:
            if (task_type == TASK_TYPE_VIDEO_LIST) or (task_type == TASK_TYPE_YOUTUBE_LIST):
                errors_per_video += 1
            #     video_source_idx += 1
            #     processed_frames_count = 0
            #     saved_frames_count = 0
            #     saved_objects_count = 0
            #     i = 0
            if cap.isOpened():
                cap.release()
            cap = None
            continue

        if (frame.shape[0] * frame.shape[1] * frame.shape[2]) > MAX_FRAME_MEMORY_SIZE:
            new_width = 1920
            new_height = int(new_width * frame.shape[0] / frame.shape[1])
            frame = cv2.resize(frame, (new_width, new_height), cv2.INTER_CUBIC)

        i += 1

        if (task_type != TASK_TYPE_IMAGES_FOLDER) and ((i + frame_shift) % FRAME_STEP != 0):
            continue

        frame_shift = random.randint(0, FRAME_STEP - 1)

        gc.collect()
        torch.cuda.empty_cache()

        processed_frames_count += 1

        frame_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

        #heads_boxes_mask = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)

        heads_bboxes = []
        heads_bboxes_to_parse = []
        heads_bboxes_to_parse_mask_detected = []
        heads_bboxes_to_parse_classes = []

        frame_has_heads = False

        #frame = cv2.resize(frame, (1280, 720))

        # if HEAD_DETECTOR == HEADS_DETECTOR_HEADS:
        img = frame
        # else:
        #     img = frame.clone()

        if HEADS_DETECTOR_PRECONVERT_GRAY:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        with torch.no_grad():
            # first_batch = eval_network(transform_frame(get_next_frame(vid)))
            frame_width = img.shape[1]
            frame_height = img.shape[0]
            #frames = [img.copy()]
            frames = [img]
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            imgs = yolact_transform(torch.stack(frames, 0))
            num_extra = 0
            while imgs.size(0) < yolact_video_multiframe:
                imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
                num_extra += 1

    #    with torch.no_grad():
            heads_out = heads_yolact_net(imgs)
            if num_extra > 0:
                heads_out = heads_out[:-num_extra]
            heads_first_batch = frames, heads_out
            heads_frames = [{'value': yolact_extract_frame(heads_first_batch, 0), 'idx': 0}]
            # def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
            heads_img = heads_frames[0]['value'][0]
            h = frame_height
            w = frame_width
            # undo_transform = True
            undo_transform = False
            class_color = False
            mask_alpha = 0.45
            """
            Note: If undo_transform=False then im_h and im_w are allowed to be None.
            """
            if undo_transform:
                heads_img_numpy = yolact_undo_image_transformation(heads_img, w, h)
                # img_numpy = rgb_imgNormal
                heads_img_gpu = torch.Tensor(heads_img_numpy).cuda()
            else:
                heads_img_gpu = heads_img / 255.0
                h, w, _ = heads_img.shape
            heads_save = heads_yolact_cfg.rescore_bbox
            heads_yolact_cfg.rescore_bbox = True
            heads_t = heads_yolact_postprocess(heads_out, w, h, visualize_lincomb=yolact_display_lincomb,
                                         crop_masks=yolact_crop,
                                         score_threshold=heads_yolact_score_threshold)
            heads_yolact_cfg.rescore_bbox = heads_save
            heads_idx = heads_t[1].argsort(0, descending=True)[:yolact_top_k]
            if heads_yolact_cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                heads_masks = heads_t[3][heads_idx]
            heads_classes, heads_scores, heads_boxes = [x[heads_idx].cpu().numpy() for x in heads_t[:3]]
            heads_num_dets_to_consider = min(yolact_top_k, heads_classes.shape[0])
            for j in range(heads_num_dets_to_consider):
                if heads_scores[j] < heads_yolact_score_threshold:
                    heads_num_dets_to_consider = j
                    break


            # Quick and dirty lambda for selecting the color for a particular index
            # Also keeps track of a per-gpu color cache for maximum speed
            '''def get_color(j, on_gpu=None):
                heads_color_idx = (heads_classes[j] * 5 if class_color else j * 5) % len(yolact_COLORS)

                if on_gpu is not None and heads_color_idx in yolact_color_cache[on_gpu]:
                    return yolact_color_cache[on_gpu][heads_color_idx]
                else:
                    heads_color = yolact_COLORS[heads_color_idx]
                    if not undo_transform:
                        # The image might come in as RGB or BRG, depending
                        heads_color = (heads_color[2], heads_color[1], heads_color[0])
                    if on_gpu is not None:
                        heads_color = torch.Tensor(heads_color).to(on_gpu).float() / 255.
                        yolact_color_cache[on_gpu][heads_color_idx] = heads_color
                    return heads_color'''


            # First, draw the masks on the GPU where we can do it really fast
            # Beware: very fast but possibly unintelligible mask-drawing code ahead
            # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
            '''if yolact_display_masks and yolact_cfg_heads.eval_mask_branch and num_dets_to_consider > 0:
                # After this, mask is of size [num_dets, h, w, 1]
                masks = masks[:num_dets_to_consider, :, :, None]
    
                # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
                colors = torch.cat(
                    [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in
                     range(num_dets_to_consider)],
                    dim=0)
                masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
                # This is 1 everywhere except for 1-mask_alpha where the mask is
                inv_alph_masks = masks * (-mask_alpha) + 1
                # I did the math for this on pen and paper. This whole block should be equivalent to:
                #    for j in range(num_dets_to_consider):
                #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
                masks_color_summand = masks_color[0]
                if num_dets_to_consider > 1:
                    inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
                    masks_color_cumul = masks_color[1:] * inv_alph_cumul
                    masks_color_summand += masks_color_cumul.sum(dim=0)
                img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand'''
            # Then draw the stuff that needs to be done on the cpu
            # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
            heads_img_numpy = (heads_img_gpu * 255).byte().cpu().numpy()
            if heads_num_dets_to_consider > 0:
                if yolact_display_text or yolact_display_bboxes:
                    for j in reversed(range(heads_num_dets_to_consider)):
                        head_class = heads_classes[j]
                        if (HEAD_DETECTOR == HEADS_DETECTOR_FACEMASKNOSES) and (head_class != 2):
                            continue
                        head_x1, head_y1, head_x2, head_y2 = heads_boxes[j, :]
                        head_h = head_y2 - head_y1
                        head_w = head_x2 - head_x1
                        if head_w <= HEAD_SIZE_MIN:
                            continue # +2
                        head_xc = (head_x2 + head_x1) / 2
                        head_yc = (head_y2 + head_y1) / 2
                        head_score = heads_scores[j]
                        mask_detected = False
                        ignore = False
                        for k in reversed(range(heads_num_dets_to_consider)):
                            if k == j:
                                continue
                            head_k_class = heads_classes[k]
                            if (HEAD_DETECTOR == HEADS_DETECTOR_FACEMASKNOSES) and (head_k_class != 2):
                                continue
                            head_k_score = heads_scores[k]
                            head_k_x1, head_k_y1, head_k_x2, head_k_y2 = heads_boxes[k, :]
                            head_k_h = head_k_y2 - head_k_y1
                            head_k_w = head_k_x2 - head_k_x1
                            if head_k_w <= HEAD_SIZE_MIN:
                                continue # +12 +10 +5 +11 +5 +3
                            head_k_xc = (head_k_x2 + head_k_x1) / 2
                            head_k_yc = (head_k_y2 + head_k_y1) / 2
                            if ((head_x1 < head_k_xc < head_x2) and (head_y1 < head_k_yc < head_y2)) or ((head_k_x1 < head_xc < head_k_x2) and (head_k_y1 < head_yc < head_k_y2)):
                                if head_score <= head_k_score:
                                    ignore = True
                                    break
                        if ignore:  # +68
                            # pass
                            continue

                        if HEAD_DETECTOR == HEADS_DETECTOR_FACEMASKNOSES:
                            nose_found = False
                            nose_in_mask = False
                            face_area = head_w * head_h
                            for idx_nose in reversed(range(heads_num_dets_to_consider)):
                                if heads_classes[idx_nose] != 0:
                                    continue
                                nose_score = heads_scores[idx_nose]
                                nose_x1, nose_y1, nose_x2, nose_y2 = heads_boxes[idx_nose, :]
                                nose_w = nose_x2 - nose_x1
                                nose_h = nose_y2 - nose_y1
                                nose_area = nose_w * nose_h
                                if nose_area > face_area / 6:
                                    continue # +25 +24 +16 +18 +15 +16
                                nose_xc = (nose_x2 + nose_x1) / 2
                                nose_yc = (nose_y2 + nose_y1) / 2
                                if (nose_xc < head_x1) or (nose_xc > head_x2) or (nose_yc < head_y1) or (nose_yc > head_y2):
                                    continue # +
                                nose_found = True
                                this_nose_in_mask = False
                                for idx_mask in reversed(range(heads_num_dets_to_consider)):
                                    if heads_classes[idx_mask] != 1:
                                        continue
                                    mask_score = heads_scores[idx_mask]
                                    mask_mask_gpu = heads_masks[idx_mask]
                                    mask_x1, mask_y1, mask_x2, mask_y2 = heads_boxes[idx_mask, :]
                                    if (mask_mask_gpu[int(nose_yc), int(nose_xc)] > 0):
                                        this_nose_in_mask = True
                                        break
                                if this_nose_in_mask:
                                    nose_in_mask = True
                                    break
                            if (not nose_found) or nose_in_mask:
                                mask_detected = True
                            #else:
                            #    mask_detected = False
                        elif HEAD_DETECTOR == HEADS_DETECTOR_MASKFACES:
                            if head_class == 0:
                                mask_detected = True
                            #else:
                            #    mask_detected = False
                        elif HEAD_DETECTOR == HEADS_DETECTOR_HEADS:
                            pass
                        #if (heads_classes[j] == 0) and ((head_x2 - head_x1) > HEAD_SIZE_MIN):

                        if (head_y1 > 0) and HEAD_USE_HEIGHT_CORRECTION:
                            head_h = int(max(head_h, head_w * 4 / 3))
                            # head_h = int(max(head_h, head_w * 3 / 2))
                            # head_h = int(max(head_h, head_w * 4 / 2))
                            head_y2 = int(head_y1 + head_h)
                        # y_min = int(max(0, head_y1 - head_w / 1))
                        ##y_max = int((y1 + y2) / 1.5)
                        # y_max = head_y2
                        # x_min = int(max(0, head_x1 - head_w / 0.7))
                        # x_max = int(min(frame.shape[1], head_x2 + head_w / 0.7))
                        '''sub_frame = frame[y_min:y_max, x_min:x_max]
                        sub_frame = cv2.resize(sub_frame, (sub_frame.shape[1] * 3, sub_frame.shape[0] * 3))
                        #gb = cv2.GaussianBlur(sub_frame, (0, 0), 3)
                        #sub_frame = cv2.addWeighted(sub_frame, 1.5, gb, -0.5, 0)
                        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        sub_frame = cv2.filter2D(sub_frame, -1, kernel)
                        bboxesCompressed = faced_face_detector.predict(sub_frame, 0.8)
                        for xc, yc, w, h, p in bboxesCompressed:
                            rx1 = int(xc - w / 2)
                            ry1 = int(yc - h / 2)
                            rx2 = int(xc + w / 2)
                            ry2 = int(yc + h / 2)
                            rx_min = int(max(0, rx1 - (rx2 - rx1) / 2))
                            ry_min = int(max(0, ry1 - (rx2 - rx1) / 2))
                            rx_max = int(min(frame.shape[1], rx2 + (rx2 - rx1) / 2))
                            cv2.rectangle(sub_frame, (rx_min, ry_min), (rx_max, ry2), (0, 255, 0), 1)'''
                        #cv2.imshow('frame1', sub_frame)
                        #cv2.rectangle(heads_img_numpy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        #cv2.rectangle(heads_boxes_mask, (x_min, y_min), (x_max, y_max), (255, 255, 255), cv2.FILLED)
                        #head_bboxes.append((x_min, y_min, x_max, y_max))

                        head_center_x = (head_x1 + head_x2) / 2
                        head_center_y = (head_y1 + head_y2) / 2
                        ignore_this_head = False
                        for head_compare_x1, head_compare_y1, head_compare_x2, head_compare_y2 in heads_bboxes_to_parse:
                            # head_compare_h = head_compare_y2 - head_compare_y1
                            # head_compare_w = head_compare_x2 - head_compare_x1
                            # if (head_compare_y1 > 0) and HEAD_USE_HEIGHT_CORRECTION:
                            #     head_compare_h = max(head_compare_h, head_compare_w * 4 / 3)
                            #     head_compare_y2 = int(head_compare_y1 + head_compare_h)
                            if (head_compare_x1 < head_center_x < head_compare_x2) and (head_compare_y1 < head_center_y < head_compare_y2):
                                ignore_this_head = True
                        if ignore_this_head:
                            # pass
                            continue # ? +62
                        head_exists_in_last = False
                        if person_track_filtering:
                            for last_head_bbox in last_heads_bboxes:
                                last_head_x1, last_head_y1, last_head_x2, last_head_y2 = last_head_bbox
                                # last_head_h = last_head_y2 - last_head_y1
                                # last_head_w = last_head_x2 - last_head_x1
                                # if (last_head_y1 > 0) and HEAD_USE_HEIGHT_CORRECTION:
                                #     last_head_h = max(last_head_h, last_head_w * 4 / 3)
                                #     last_head_y2 = int(last_head_y1 + last_head_h)
                                if (last_head_x1 < head_center_x < last_head_x2) and (last_head_y1 < head_center_y < last_head_y2):
                                    head_exists_in_last = True
                        if (not head_exists_in_last) or (random.randint(0, 9) == 0):
                            heads_bboxes_to_parse.append((head_x1, head_y1, head_x2, head_y2))
                            heads_bboxes_to_parse_mask_detected.append(mask_detected)
                            heads_bboxes_to_parse_classes.append(head_class)
                        heads_bboxes.append((head_x1, head_y1, head_x2, head_y2))

                        # head_color = get_color(j)
                        # head_score = heads_scores[j]
                        '''if yolact_display_bboxes:
                            cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
                        if yolact_display_text:
                            _class = yolact_cfg_heads.dataset.class_names[classes[j]]
                            text_str = '%s: %.2f' % (_class, score) if yolact_display_scores else _class
                            font_face = cv2.FONT_HERSHEY_DUPLEX
                            font_scale = 0.6
                            font_thickness = 1
                            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                            text_pt = (x1, y1 - 3)
                            text_color = [255, 255, 255]
                            cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                            cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                                        cv2.LINE_AA)'''

        #cv2.imshow('heads', img_numpy_heads)

        #cv2.imshow('head_boxes_mask', head_boxes_mask)

        #img_numpy_heads = cv2.cvtColor(img_numpy_heads, cv2.COLOR_BGR2RGB)
        #img_numpy_heads = cv2.bitwise_not(img_numpy_heads)
        #img_gpu_heads = torch.Tensor(img_numpy_heads).cuda() / 255.0

        last_heads_bboxes = heads_bboxes







        #people_mask = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
        people_mask_gpu = torch.zeros(frame.shape[0], frame.shape[1], 3).cuda()

        #img = frame
        with torch.no_grad():
            '''# first_batch = eval_network(transform_frame(get_next_frame(vid)))
            frame_width = img.shape[1]
            frame_height = img.shape[0]
            frames = [img.copy()]
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            imgs = yolact_transform(torch.stack(frames, 0))
            num_extra = 0
            while imgs.size(0) < yolact_video_multiframe:
                imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
                num_extra += 1'''
            people_out = people_yolact_net(imgs)
            if num_extra > 0:
                people_out = people_out[:-num_extra]
            people_first_batch = frames, people_out
            people_frames = [{'value': yolact_extract_frame(people_first_batch, 0), 'idx': 0}]
            # def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
            # img = frames
            people_img = people_frames[0]['value'][0]
            h = frame_height
            w = frame_width
            # undo_transform = True
            undo_transform = False
            class_color = False
            mask_alpha = 0.45
            """
            Note: If undo_transform=False then im_h and im_w are allowed to be None.
            """
            if undo_transform:
                people_img_numpy = yolact_undo_image_transformation(people_img, w, h)
                # img_numpy = rgb_imgNormal
                people_img_gpu = torch.Tensor(2).cuda()
            else:
                people_img_gpu = people_img / 255.0
                #img_gpu_people = le_mask_gpu / 255.0
                h, w, _ = people_img.shape
            people_save = people_yolact_cfg.rescore_bbox
            people_yolact_cfg.rescore_bbox = True
            people_t = people_yolact_postprocess(people_out, w, h, visualize_lincomb=yolact_display_lincomb,
                                          crop_masks=yolact_crop,
                                          score_threshold=people_yolact_score_threshold)
            people_yolact_cfg.rescore_bbox = people_save
            people_idx = people_t[1].argsort(0, descending=True)[:yolact_top_k]
            if people_yolact_cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                people_masks = people_t[3][people_idx]
            people_classes, people_scores, people_boxes = [x[people_idx].cpu().numpy() for x in people_t[:3]]
            people_num_dets_to_consider = min(yolact_top_k, people_classes.shape[0])
            for j in range(people_num_dets_to_consider):
                if people_scores[j] < people_yolact_score_threshold:
                    people_num_dets_to_consider = j
                    break


            # Quick and dirty lambda for selecting the color for a particular index
            # Also keeps track of a per-gpu color cache for maximum speed
            '''def get_color(j, on_gpu=None):
                people_color_idx = (people_classes[j] * 5 if class_color else j * 5) % len(yolact_COLORS)

                if on_gpu is not None and people_color_idx in yolact_color_cache[on_gpu]:
                    return yolact_color_cache[on_gpu][people_color_idx]
                else:
                    people_color = yolact_COLORS[people_color_idx]
                    if not undo_transform:
                        # The image might come in as RGB or BRG, depending
                        people_color = (people_color[2], people_color[1], people_color[0])
                    if on_gpu is not None:
                        people_color = torch.Tensor(people_color).to(on_gpu).float() / 255.
                        yolact_color_cache[on_gpu][people_color_idx] = people_color
                    return people_color'''


            # First, draw the masks on the GPU where we can do it really fast
            # Beware: very fast but possibly unintelligible mask-drawing code ahead
            # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
            if yolact_display_masks and people_yolact_cfg.eval_mask_branch and people_num_dets_to_consider > 0:
                # After this, mask is of size [num_dets, h, w, 1]
                people_masks = people_masks[:people_num_dets_to_consider, :, :, None]

                ''' # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
                colors = torch.cat(
                    [get_color(j, on_gpu=people_img_gpu.device.index).view(1, 1, 1, 3) for j in
                     range(people_num_dets_to_consider)],
                    dim=0)
                people_masks_color = people_masks.repeat(1, 1, 1, 3) * colors * mask_alpha
                # This is 1 everywhere except for 1-mask_alpha where the mask is
                people_inv_alph_masks = people_masks * (-mask_alpha) + 1
                # I did the math for this on pen and paper. This whole block should be equivalent to:
                #    for j in range(num_dets_to_consider):
                #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
                people_masks_color_summand = people_masks_color[0]
                if people_num_dets_to_consider > 1:
                    people_inv_alph_cumul = people_inv_alph_masks[:(people_num_dets_to_consider - 1)].cumprod(dim=0)
                    people_masks_color_cumul = people_masks_color[1:] * people_inv_alph_cumul
                    people_masks_color_summand += people_masks_color_cumul.sum(dim=0)
                people_img_gpu = people_img_gpu * people_inv_alph_masks.prod(dim=0) + people_masks_color_summand'''

                #masks_cpu = (masks * 255).byte().cpu().numpy()
            # Then draw the stuff that needs to be done on the cpu
            # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
            #img_numpy_people = (img_gpu_people * 255).byte().cpu().numpy()
            #people_img_numpy = (people_img_gpu * 255).byte().cpu().numpy()
            people_img_numpy = frame.copy()
            #people_img_numpy = frame
            if people_num_dets_to_consider > 0:
                if yolact_display_text or yolact_display_bboxes:
                    for j in reversed(range(people_num_dets_to_consider)):
                        person_x1, person_y1, person_x2, person_y2 = people_boxes[j, :]
                        person_w = person_x2 - person_x1
                        person_h = person_y2 - person_y1
                        if (people_classes[j] != 0) or (person_w < HEAD_SIZE_MIN * 1.5):
                            continue # +
                        person_score = people_scores[j]
                        person_xc = int((person_x1 + person_x2) / 2)
                        person_yc = int((person_y1 + person_y2) / 2)

                        '''ignore_person = False
                        for k in reversed(range(people_num_dets_to_consider)):
                            person_k_x1, person_k_y1, person_k_x2, person_k_y2 = people_boxes[k, :]
                            person_k_w = person_k_x2 - person_k_x1
                            person_k_h = person_k_y2 - person_k_y1
                            if (people_classes[k] != 0) or (person_k_w < HEAD_SIZE_MIN * 1.5):
                                continue
                            person_k_score = people_scores[k]
                            person_k_xc = int((person_k_x1 + person_k_x2) / 2)
                            person_k_yc = int((person_k_y1 + person_k_y2) / 2)
                            if ((person_k_x1 < person_xc < person_k_x2) and (person_k_y1 < person_yc < person_k_y2)) or ((person_x1 < person_k_xc < person_x2) and (person_y1 < person_k_yc < person_y2)):
                                # if (person_k_score > person_score) and (person_k_w * person_k_h > person_w * person_h):
                                # if person_k_score > person_score:
                                if person_k_w * person_k_h > person_w * person_h:
                                    ignore_person = True
                        if ignore_person:
                            continue'''

                        '''person_y_min = int(max(0, person_y1 - (person_x2 - person_x1) / 2))
                        person_y_max = int((person_y1 + person_y2) / 1.5)
                        person_x_min = int(max(0, person_x1 - (person_x2 - person_x1) / 1))
                        person_x_max = int(min(frame.shape[1], person_x2 + (person_x2 - person_x1) / 1))
                        sub_frame = frame[person_y_min:person_y_max, person_x_min:person_x_max]
                        sub_frame = cv2.resize(sub_frame, (sub_frame.shape[1] * 3, sub_frame.shape[0] * 3))'''
                        '''#gb = cv2.GaussianBlur(sub_frame, (0, 0), 3)
                        #sub_frame = cv2.addWeighted(sub_frame, 1.5, gb, -0.5, 0)
                        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        sub_frame = cv2.filter2D(sub_frame, -1, kernel)
                        bboxesCompressed = faced_face_detector.predict(sub_frame, 0.8)
                        for xc, yc, w, h, p in bboxesCompressed:
                            rx1 = int(xc - w / 2)
                            ry1 = int(yc - h / 2)
                            rx2 = int(xc + w / 2)
                            ry2 = int(yc + h / 2)
                            rx_min = int(max(0, rx1 - (rx2 - rx1) / 2))
                            ry_min = int(max(0, ry1 - (rx2 - rx1) / 2))
                            rx_max = int(min(frame.shape[1], rx2 + (rx2 - rx1) / 2))
                            cv2.rectangle(sub_frame, (rx_min, ry_min), (rx_max, ry2), (0, 255, 0), 1)'''
                        #cv2.imshow('frame1', sub_frame)
                        people_mask_cpu = (people_masks[j] * 255).byte().cpu().numpy()
                        #if not mask_cpu.any():
                        #if np.count_nonzero(mask_cpu) > 100:
                        for head_idx, head_bbox in enumerate(heads_bboxes_to_parse):
                            head_x1, head_y1, head_x2, head_y2 = head_bbox
                            head_w = head_x2 - head_x1
                            head_h = head_y2 - head_y1
                            # if (head_y1 > 0) and HEAD_USE_HEIGHT_CORRECTION2:
                            #     head_h = int(head_h * 1.5)
                            #     head_y2 = int(head_y1 + head_h)
                            head_center_x = int((head_x2 + head_x1) / 2)
                            head_center_y = int((head_y2 + head_y1) / 2)
                            # if not ((person_x1 < head_center_x < person_x2) and (person_y1 < head_center_y < person_y2)):
                            #     continue # +++many
                            if people_mask_cpu[head_center_y, head_center_x] == 0:
                                continue # +++many
                            cv2.rectangle(people_img_numpy, (head_x1, head_y1), (head_x2, head_y2), (0, 255, 0), 2)
                            head_y_min = int(max(0, head_y1 - head_w / 1))
                            # head_y_max = int((y1 + y2) / 1.5)
                            if HEAD_USE_HEIGHT_CORRECTION2:
                                head_y_max = int(min(frame.shape[0], head_y2 + head_h / 4))
                            else:
                                head_y_max = head_y2
                            head_x_min = int(max(0, head_x1 - head_w / 0.7))
                            head_x_max = int(min(frame.shape[1], head_x2 + head_w / 0.7))
                            cv2.rectangle(people_img_numpy, (head_x_min, head_y_min), (head_x_max, head_y_max), (255, 0, 0), 2)
                            if SAVE_PEOPLE:
                                person_mask = people_mask_cpu[person_y1:person_y2, person_x1:person_x2].copy()
                                person_frame = frame[person_y1:person_y2, person_x1:person_x2].copy()
                            roi_mask = people_mask_cpu[head_y_min:head_y_max, head_x_min:head_x_max].copy()
                            #roi_frame = frame[head_y_min:head_y_max, head_x_min:head_x_max].copy()
                            # cv2.imshow("person_frame", frame[head_y1:head_y2, head_x1:head_x2])
                            # cv2.waitKey(0)
                            def findMaskRowStartEnd(row):
                                nonzero_x_start = 0
                                nonzero_x_start_is_set = False
                                nonzero_x_end = row.shape[0] - 1
                                nonzero_x_end_is_set = False
                                for iw in range(row.shape[0]):
                                    if (not nonzero_x_start_is_set) and (row[iw][0] > 0):
                                        nonzero_x_start = iw
                                        nonzero_x_start_is_set = True
                                    if (not nonzero_x_end_is_set) and (row[row.shape[0] - iw - 1][0] > 0):
                                        nonzero_x_end = row.shape[0] - iw - 1
                                        nonzero_x_end_is_set = True
                                    if nonzero_x_start_is_set and nonzero_x_end_is_set:
                                        break
                                return nonzero_x_start, nonzero_x_end

                            top_nonzeros = 0
                            if HEAD_RECTIFY_RATIO >= 1:
                                nonzeros_last = 0
                                nonzeros_dec_amount = 0
                                nonzeros_inc_amount = 0
                                head_y_max_adjusted = head_y_max
                                #nonzero_x_start = 0
                                #nonzero_x_start_is_set = False
                                #nonzero_x_end = roi_mask.shape[1] - 1
                                #nonzero_x_end_is_set = False
                                for ih in range(roi_mask.shape[0]):
                                    roi_row = roi_mask[roi_mask.shape[0] - ih - 1]
                                    nonzeros = np.count_nonzero(roi_row)
                                    if nonzeros_last == 0:
                                        nonzeros_last = nonzeros
                                        continue
                                    nonzeros_change = nonzeros_last - nonzeros
                                    if nonzeros <= nonzeros_last:
                                        nonzeros_dec_amount += nonzeros_change
                                    else:
                                        nonzeros_inc_amount += -nonzeros_change
                                    nonzeros_last = nonzeros
                                    '''if (nonzeros_dec_amount > 3) and (nonzeros_inc_amount > 3):
                                        head_y_max_adjusted = head_y_max - ih
                                        break'''
                                    '''if nonzeros_inc_amount > 0:
                                        head_y_max_adjusted = head_y_max - ih
                                        break'''
                                    if nonzeros_change == 0:
                                        head_y_max_adjusted = head_y_max - ih
                                        nonzero_x_start, nonzero_x_end = findMaskRowStartEnd(roi_row)
                                        break
                                    if (nonzeros_dec_amount <= 3) and (nonzeros_inc_amount > 3):
                                        break
                                if (head_y_max_adjusted - head_y_max) != 0:
                                    #roi_mask = cv2.rectangle(roi_mask, (0, roi_frame.shape[0] + head_y_max_adjusted - head_y_max), (nonzero_x_start, roi_frame.shape[0]), (0), cv2.FILLED)
                                    #roi_mask = cv2.rectangle(roi_mask, (nonzero_x_end + 1, roi_frame.shape[0] + head_y_max_adjusted - head_y_max), (roi_frame.shape[1], roi_frame.shape[0]), (0), cv2.FILLED)
                                    roi_mask = cv2.rectangle(roi_mask, (0, head_y_max_adjusted - head_y_min), (nonzero_x_start, head_y_max - head_y_min), (0), cv2.FILLED)
                                    roi_mask = cv2.rectangle(roi_mask, (nonzero_x_end + 1, head_y_max_adjusted + head_y_min), (head_x_max - head_x_min, head_y_max - head_y_min), (0), cv2.FILLED)
                            else:  #HEAD_RECTIFY_RATIO < 1:
                                top_nonzeros_ih = -1
                                nonzero_x_rectify_start, nonzero_x_rectify_end = 0, roi_mask.shape[1] - 1
                                for ih in range(roi_mask.shape[0]):
                                    roi_row = roi_mask[ih]
                                    nonzeros = np.count_nonzero(roi_row)
                                    if nonzeros == 0:
                                        continue
                                    if top_nonzeros_ih == -1:
                                        top_nonzeros = nonzeros
                                        top_nonzeros_ih = ih
                                        head_y_rectify_start = top_nonzeros_ih + int((roi_mask.shape[0] - top_nonzeros_ih) * HEAD_RECTIFY_RATIO)
                                    if ih < head_y_rectify_start:
                                        continue
                                    elif ih == head_y_rectify_start:
                                        nonzero_x_rectify_start, nonzero_x_rectify_end = findMaskRowStartEnd(roi_row)
                                    else:
                                        nonzero_x_start, nonzero_x_end = findMaskRowStartEnd(roi_row)
                                        nonzero_x_rectify_start = max(nonzero_x_start, nonzero_x_rectify_start)
                                        nonzero_x_rectify_end = min(nonzero_x_end, nonzero_x_rectify_end)
                                        roi_row[:nonzero_x_rectify_start] = 0
                                        roi_row[nonzero_x_rectify_start:nonzero_x_rectify_end] = 255
                                        roi_row[nonzero_x_rectify_end:] = 0

                            #head_y_max_adjusted = int(head_y_max / 2)
                            roi_mask_gray = roi_mask # cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                            _, head_roi_binary = cv2.threshold(roi_mask_gray, 225, 255, cv2.THRESH_BINARY)
                            head_contours, hierarchy = cv2.findContours(head_roi_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            if len(head_contours) == 0:
                                continue # +
                            #peri = cv2.arcLength(c, True)
                            #approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                            #epsilon = 0.1 * cv.arcLength(cnt, True)
                            #approx = cv.approxPolyDP(cnt, epsilon, True)
                            preview_timestamp = datetime.now().strftime("%f")
                            score = str(int(round(people_scores[j] * 100, 0)))
                            max_contour = max(head_contours, key=cv2.contourArea)
                            area = cv2.contourArea(max_contour)
                            if area < HEAD_SIZE_MIN**2:
                                continue # ++







                            if INTERSECT_WITH_ELLIPSE:
                                try:
                                    #roi_mask = people_mask_cpu[head_y_min:head_y_max, head_x_min:head_x_max].copy()
                                    ellipse_mask = np.zeros_like(roi_mask_gray)
                                    ellipse = cv2.fitEllipse(max_contour)
                                    cv2.ellipse(ellipse_mask, ellipse, (255), -1)
                                    #_, ellipse_mask_binary = cv2.threshold(ellipse_mask, 225, 255, cv2.THRESH_BINARY)
                                    ellipse_mask_binary = ellipse_mask.astype(np.float) / 255
                                    roi_mask_gray = roi_mask_gray * ellipse_mask_binary.reshape(roi_mask_gray.shape)
                                    _, head_roi_binary = cv2.threshold(roi_mask_gray, 225, 255, cv2.THRESH_BINARY)
                                    head_contours, hierarchy = cv2.findContours(head_roi_binary.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                    if len(head_contours) == 0:
                                        continue # +
                                    max_contour = max(head_contours, key=cv2.contourArea)
                                except:
                                    pass

                            #scale_factor = people_img_numpy.shape[1] / head_roi_binary.shape[1]

                            #original_max_contour = (max_contour * scale_factor).astype(np.int32)
                            original_max_contour = max_contour + [head_x_min, head_y_min]

                            #cv2.drawContours(people_img_numpy, [original_max_contour], -1, (0, 255, 0), 2)
                            original_hull = cv2.convexHull(original_max_contour)

                            area = cv2.contourArea(original_max_contour)
                            hull_area = cv2.contourArea(original_hull)
                            if (area == 0) or (hull_area == 0):
                                continue
                            ratio_area_original_to_hull = area / hull_area
                            if ratio_area_original_to_hull < MIN_RATIO_AREA_ORIGINAL_TO_HULL:
                                # pass
                                continue # +++++++++++++++++++++

                            original_perimeter = cv2.arcLength(original_max_contour, True)
                            hull_perimeter = cv2.arcLength(original_hull, True)
                            ratio_perimeter_hull_to_original = hull_perimeter / original_perimeter
                            if ratio_perimeter_hull_to_original < MIN_RATIO_PERIMETER_HULL_TO_ORIGINAL:
                                pass
                                # continue # ? +++++++++++++++

                            original_min_rect = cv2.minAreaRect(original_max_contour)
                            original_min_rect_box = cv2.boxPoints(original_min_rect)
                            original_min_rect_box = np.int0(original_min_rect_box)
                            original_min_rect_width = sqrt((original_min_rect_box[1, 0] - original_min_rect_box[0, 0])**2 + (original_min_rect_box[1, 1] - original_min_rect_box[0, 1])**2)
                            original_min_rect_height = sqrt((original_min_rect_box[2, 0] - original_min_rect_box[1, 0])**2 + (original_min_rect_box[2, 1] - original_min_rect_box[1, 1])**2)
                            if (original_min_rect_height == 0) or (original_min_rect_width == 0):
                                continue
                            ratio_original_min_rect_proportion = min(original_min_rect_width / original_min_rect_height, original_min_rect_height / original_min_rect_width)
                            if ratio_original_min_rect_proportion < MIN_RATIO_ORIGINAL_MIN_RECT_PROPORTION:
                                # pass
                                continue # ++++++

                            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(max_contour)
                            if (top_nonzeros / bbox_w) > MAX_TOP_NONZEROS_RATIO:
                                pass
                                # continue # ++++++++++++++++++++

                            if SORT_BY_MASK_DETECTION:
                                if SORT_BY_MASK_DETECTION_MULTICLASS:
                                    if SORT_BY_MASK_DETECTION_MULTICLASS_EXTRA:
                                        img_roi = (frame[head_y_min:head_y_max, head_x_min:head_x_max]).copy()
                                        mask_roi = np.zeros(img_roi.shape[:2], np.uint8)
                                        cv2.drawContours(mask_roi, [max_contour], -1, (255), thickness=-1)
                                        res_roi = cv2.bitwise_and(img_roi, img_roi, mask=mask_roi)
                                        # object_image = res_roi
                                        mask_roi -= 255
                                        gray_roi = np.zeros(img_roi.shape, np.uint8)
                                        gray_roi.fill(127)
                                        gray_roi_reversed = cv2.bitwise_and(gray_roi, gray_roi, mask=mask_roi)
                                        res_roi = cv2.addWeighted(res_roi, 1, gray_roi_reversed, 1, 0)
                                        bound_rect = cv2.boundingRect(max_contour)
                                        res_roi = res_roi[bound_rect[1]:bound_rect[1] + bound_rect[3], bound_rect[0]:bound_rect[0] + bound_rect[2]]
                                        res_roi = cv2.resize(res_roi, (MULTICLASS_EXTRA_FACE_WIDTH, MULTICLASS_EXTRA_FACE_HEIGHT), cv2.INTER_CUBIC)
                                        res_roi = res_roi.reshape((1,) + res_roi.shape)
                                        # res_roi = res_roi.copy()
                                        # cv2.imshow('res_roi', res_roi.copy())
                                        predictions = multiclass_extra_mask_model.predict(res_roi)
                                        idx_max = np.argmax(predictions[0])
                                        prob_max = predictions[0][idx_max]
                                        class_name = CLASS_NAMES[idx_max]
                                        if prob_max < MULTICLASS_EXTRA_MINIMUM_ACCURACY:
                                            continue
                                        det_head_class = idx_max
                                    elif HEAD_DETECTOR == HEADS_DETECTOR_MASKFACES:
                                        det_head_class = heads_bboxes_to_parse_classes[head_idx] + 1
                                    else:
                                        det_head_class = 1
                                    subfolder = CLASS_NAMES[det_head_class] + '/'
                                else:
                                    head_mask_detected = heads_bboxes_to_parse_mask_detected[head_idx]
                                    subfolder = "mask/" if head_mask_detected else "nomask/"
                            else:
                                subfolder = ""

                            people_img_numpy2 = frame.copy()

                            cv2.drawContours(people_img_numpy2, [original_hull], -1, (0, 0, 255), 2)
                            cv2.drawContours(people_img_numpy2, [original_min_rect_box], 0, (0, 255, 255), 2)

                            # original_min_rect_M = cv2.moments(original_min_rect_box)
                            # original_min_rect_mcx = int(original_min_rect_M['m10'] / original_min_rect_M['m00'])
                            # original_min_rect_mcy = int(original_min_rect_M['m01'] / original_min_rect_M['m00'])
                            # #cv2.circle(people_img_numpy2, (original_min_rect_mcx, original_min_rect_mcy), 5, (0, 255, 255), -1)

                            try:
                                original_ellipse = cv2.fitEllipse(original_max_contour)
                                cv2.ellipse(people_img_numpy2, original_ellipse, (255, 0, 255), 2)
                            except:
                                pass

                            #centerE = original_ellipse[0]
                            # Gets rotation of ellipse; same as rotation of contour
                            #rotation = original_ellipse[2]
                            # Gets width and height of rotated ellipse
                            #widthE = original_ellipse[1][0]
                            #heightE = original_ellipse[1][1]

                            # try:
                            #     original_M = cv2.moments(original_max_contour)
                            #     original_mcx = int(original_M['m10'] / original_M['m00'])
                            #     original_mcy = int(original_M['m01'] / original_M['m00'])
                            #     #cv2.circle(people_img_numpy2, (original_mcx, original_mcy), 5, (0, 255, 0), -1)
                            # except:
                            #     pass

                            #cv2.circle(people_img_numpy2, (int(original_ellipse[0][0]), int(original_ellipse[0][1])), 5, (255, 0, 255), -1)

                            #roi_frame = people_img_numpy2[head_y_min:head_y_max, head_x_min:head_x_max].copy()
                            #roi_frame = frame[head_y_min:head_y_max, head_x_min:head_x_max].copy()
                            roi_frame = people_img_numpy2[head_y_min:head_y_max, head_x_min:head_x_max]

                            #print(area)
                            #roi = cv2.cvtColor(roi_mask_gray, cv2.COLOR_GRAY2BGR)
                            #roi = cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)
                            #cv2.imshow('masks', roi)
                            roi_frame = cv2.drawContours(roi_frame, [max_contour], -1, (0, 255, 0), 2)
                            # cv2.putText(roi_frame, 'area_o2h: ' + str(round(ratio_area_original_to_hull, 3)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (127, 127, 255), 1, cv2.LINE_AA)
                            # cv2.putText(roi_frame, 'peri_h2o: ' + str(round(ratio_perimeter_hull_to_original, 3)), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (127, 127, 255), 1, cv2.LINE_AA)
                            # cv2.putText(roi_frame, 'rect_min: ' + str(round(ratio_original_min_rect_proportion, 3)), (15, 45), cv2.FONT_HERSHEY_SIMPLEX, .5, (127, 127, 255), 1, cv2.LINE_AA)
                            #if (head_y_max_adjusted - head_y_max) != 0:
                            #    roi_frame = cv2.line(roi_frame, (0, roi_frame.shape[0] + head_y_max_adjusted - head_y_max), (roi_frame.shape[1], roi_frame.shape[0] + head_y_max_adjusted - head_y_max), (0, 255, 255), 2)
                            # cv2.imshow('head', roi_frame)
                            # cv2.waitKey(0)
                            cv2.imwrite(OBJECTS_FOLDER + subfolder + frame_timestamp + "_" + score + "_" + preview_timestamp + ".jpg", roi_frame)
                            preview_data = {'frame_ref': frame_timestamp, 'frame_width': frame.shape[1], 'frame_height': frame.shape[0], 'contour_area': int(area), 'bbox': {'x': head_x_min + bbox_x, 'y': head_y_min + bbox_y, 'w': bbox_w, 'h': bbox_h}, 'contour': []}
                            for idx, coord in enumerate(max_contour):
                                x = int(head_x_min + coord[0][0])
                                y = int(head_y_min + coord[0][1])
                                point = {'x': x, 'y': y}
                                preview_data['contour'].append(point)
                            #jstr = json.dumps(preview_data, indent=4)
                            with open(OBJECTS_FOLDER + frame_timestamp + "_" + score + "_" + preview_timestamp + ".txt", 'w') as outfile:
                                json.dump(preview_data, outfile, indent=4)
                            saved_objects_count += 1
                            frame_has_heads = True
                            if SAVE_PEOPLE:
                                _, person_roi_binary = cv2.threshold(person_mask, 225, 255, cv2.THRESH_BINARY)
                                person_contours, hierarchy = cv2.findContours(person_roi_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                if len(person_contours) > 0:
                                    max_contour = max(person_contours, key=cv2.contourArea)
                                    area = cv2.contourArea(max_contour)
                                    person_frame = cv2.drawContours(person_frame, [max_contour], -1, (0, 255, 0), 2)
                                    #cv2.imshow('person', person_frame)
                                    cv2.imwrite(OBJECTS_FOLDER + frame_timestamp + "_" + score + "_" + preview_timestamp + ".jpg", person_frame)
                                    bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(max_contour)
                                    preview_data = {'frame_ref': frame_timestamp, 'frame_width': frame.shape[1], 'frame_height': frame.shape[0], 'contour_area': int(area), 'bbox': {'x': int(person_x1) + bbox_x, 'y': int(person_y1) + bbox_y, 'w': bbox_w, 'h': bbox_h}, 'contour': []}
                                    for idx, coord in enumerate(max_contour):
                                        x = int(person_x1 + coord[0][0])
                                        y = int(person_y1 + coord[0][1])
                                        point = {'x': x, 'y': y}
                                        preview_data['contour'].append(point)
                                    # jstr = json.dumps(preview_data, indent=4)
                                    with open(OBJECTS_FOLDER + frame_timestamp + "_" + score + "_" + preview_timestamp + ".txt", 'w') as outfile:
                                        json.dump(preview_data, outfile, indent=4)

                                #cv2.imshow('masks', mask_cpu * (head_boxes_mask / 255))
                        # person_color = get_color(j)
                        # person_score = people_scores[j]
                        '''if yolact_display_bboxes:
                            #cv2.rectangle(img_numpy2, (x1, y1), (x2, y2), color, 1)
                            #cv2.rectangle(people_img_numpy, (person_x1, person_y1), (person_x2, person_y2), (255, 255, 0), 1)
                            pass'''
                        '''if yolact_display_text:
                            _class = yolact_cfg_people.dataset.class_names[classes[j]]
                            text_str = '%s: %.2f' % (_class, score) if yolact_display_scores else _class
                            font_face = cv2.FONT_HERSHEY_DUPLEX
                            font_scale = 0.6
                            font_thickness = 1
                            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                            text_pt = (x1, y1 - 3)
                            text_color = [255, 255, 255]
                            cv2.rectangle(img_numpy2, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                            cv2.putText(img_numpy2, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                                        cv2.LINE_AA)'''

        cv2.imshow('people', people_img_numpy)

        #cv2.imshow('frame', frame)

        #cv2.imshow('res', img_numpy2 * (head_boxes_mask / 255))

        if frame_has_heads:
            #cv2.imwrite(pathToScriptFolder + "/parsed_data/frames/frame_" + str(frame_idx) + "_" + frame_timestamp + ".jpg", frame)
            cv2.imwrite(FRAMES_FOLDER + frame_timestamp + ".jpg", frame)
            saved_frames_count += 1

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    if (cap is not None) and cap.isOpened():
        cap.release()

# cv2.waitKey(0)

cv2.destroyAllWindows()

