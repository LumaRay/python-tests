from backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone
from math import sqrt
import torch

# for making bounding boxes pretty
COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))


# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD   = (57.38, 57.12, 58.40)

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}



# ----------------------- CONFIG CLASS ----------------------- #

class Config(object):
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = Config(vars(self))
        
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)
    
    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)





# ----------------------- DATASETS ----------------------- #

dataset_base = Config({
    'name': 'Base Dataset',

    # Training images and annotations
    'train_images': './data/coco/images/',
    'train_info':   'path_to_annotation_file',

    # Validation images and annotations.
    'valid_images': './data/coco/images/',
    'valid_info':   'path_to_annotation_file',

    # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
    'has_gt': True,

    # A list of names for each of you classes.
    'class_names': COCO_CLASSES,

    # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
    # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
    # If not specified, this just assumes category ids start at 1 and increase sequentially.
    'label_map': None
})

coco2014_dataset = dataset_base.copy({
    'name': 'COCO 2014',
    
    'train_info': './data/coco/annotations/instances_train2014.json',
    'valid_info': './data/coco/annotations/instances_val2014.json',

    'label_map': COCO_LABEL_MAP
})

coco2017_dataset = dataset_base.copy({
    'name': 'COCO 2017',
    
    'train_info': './data/coco/annotations/instances_train2017.json',
    'valid_info': './data/coco/annotations/instances_val2017.json',

    'label_map': COCO_LABEL_MAP
})

coco2017_testdev_dataset = dataset_base.copy({
    'name': 'COCO 2017 Test-Dev',

    'valid_info': './data/coco/annotations/image_info_test-dev2017.json',
    'has_gt': False,

    'label_map': COCO_LABEL_MAP
})

PASCAL_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle",
                  "bus", "car", "cat", "chair", "cow", "diningtable",
                  "dog", "horse", "motorbike", "person", "pottedplant",
                  "sheep", "sofa", "train", "tvmonitor")

pascal_sbd_dataset = dataset_base.copy({
    'name': 'Pascal SBD 2012',

    'train_images': './data/sbd/img',
    'valid_images': './data/sbd/img',
    
    'train_info': './data/sbd/pascal_sbd_train.json',
    'valid_info': './data/sbd/pascal_sbd_val.json',

    'class_names': PASCAL_CLASSES,
})

mask_faces_dataset = dataset_base.copy({
  'name': 'Faces With Masks',
  'train_info': './data/mask_faces/train/via_project_1Nov2020_16h10m_coco.json',
  'train_images': './data/mask_faces/train/images/',
  'valid_info': './data/mask_faces/val/via_project_1Nov2020_16h10m_coco.json',
  'valid_images': './data/mask_faces/val/images/',
  'class_names': ('unknown', 'nomask', 'badmask', 'mask'),
  'label_map': { 1:  1, 2:  2, 3:  3, 4:  4 }
})

heads_dataset = dataset_base.copy({
  'name': 'Heads',
  'train_info': './data/heads/train/via_project_1Nov2020_16h10m_coco.json',
  'train_images': './data/heads/train/images/',
  'valid_info': './data/heads/val/via_project_1Nov2020_16h10m_coco.json',
  'valid_images': './data/heads/val/images/',
  'class_names': ('head'),
  'label_map': { 1:  1 }
})

headsgray_dataset = dataset_base.copy({
  'name': 'headsgray',
  'train_info': './data/heads/train/via_project_1Nov2020_16h10m_coco.json',
  'train_images': './data/heads/train/images/gray/',
  'valid_info': './data/heads/val/via_project_1Nov2020_16h10m_coco.json',
  'valid_images': './data/heads/val/images/gray/',
  'class_names': ('head'),
  'label_map': { 1:  1 }
})

headsnoses_dataset = dataset_base.copy({
  'name': 'headsnoses',
  'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/headsnoses/train/via_project_headnose_train_9Nov2020_18h51m_coco.json',
  'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/headsnoses/train/images/',
  'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/headsnoses/val/via_project_headnose_valid_9Nov2020_22h41m_coco.json',
  'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/headsnoses/val/images/',
  'class_names': ('head', 'nose'),
  'label_map': { 1: 1, 2: 2 }
})

maskfacesnew_dataset = dataset_base.copy({
    'name': 'maskfacesnew',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnew/train/2020-11-07-23-19-08-394529_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnew/train/images',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnew/valid/2020-11-07-23-19-08-394529_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnew/valid/images',
    'class_names': ('back', 'mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4, 5: 5 }
})

maskfacesnew2_dataset = dataset_base.copy({
    'name': 'maskfacesnew2',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnew2/train/2020-11-08-19-39-27-762783_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnew2/train/images',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnew2/valid/2020-11-08-19-39-27-762783_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnew2/valid/images',
    'class_names': ('mask', 'masknone'),
    'label_map': { 1: 1, 2: 2 }
})

maskfacesnew03_dataset = dataset_base.copy({
    'name': 'maskfacesnew03',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnew03/train/2020-11-17-23-57-06-005076_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnew03/train/images',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnew03/valid/2020-11-17-23-57-06-005076_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnew03/valid/images',
    'class_names': ('back', 'mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4, 5: 5 }
})

maskfacesnewwork0312mask_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312mask',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312mask/train/2020-11-21-16-18-16-767109_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312mask/train/images',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312mask/valid/2020-11-21-16-18-16-767109_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312mask/valid/images',
    'class_names': ('face', 'mask'),
    'label_map': { 1: 1, 2: 2 }
})

maskfacesnewwork0312added1nb_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1nb',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1nb/train/2020-11-26-19-59-50-143805_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1nb/train/images',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1nb/valid/2020-11-26-19-59-50-143805_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1nb/valid/images',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

noses_dataset = dataset_base.copy({
    'name': 'noses',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/noses/train/train_nose_via_project_29Nov2020_13h35m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/noses/train/images/',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/noses/val/val_nose_via_project_29Nov2020_13h35m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/noses/val/images/',
    'class_names': ('nose'),
    'label_map': { 1: 1 }
})

headsnew_dataset = dataset_base.copy({
	'name': 'headsnew',
	'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/headsnew/train/2021-01-20-22-17-41-263598_coco_train.json',
	'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/headsnew/train/images/',
	'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/headsnew/valid/2021-01-20-22-17-41-263598_coco_valid.json',
	'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/headsnew/valid/images/',
	'class_names': ('head'),
	'label_map': { 1:  1 }
})

headsnewgray_dataset = dataset_base.copy({
	'name': 'headsnewgray',
	'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/headsnew/train/2021-01-20-22-17-41-263598_coco_train.json',
	'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/headsnew/train/images/gray3',
	'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/headsnew/valid/2021-01-20-22-17-41-263598_coco_valid.json',
	'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/headsnew/valid/images/gray3',
	'class_names': ('head'),
	'label_map': { 1:  1 }
})

facemasknoses_dataset = dataset_base.copy({
    'name': 'facemasknoses',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesextblur9blur3blur1sharp9bright10dark10cont10decont10decont6sat10desat10normdevdevvtest1_dataset = dataset_base.copy({
    'name': 'facemasknosesextblur9blur3blur1sharp9bright10dark10cont10decont10decont6sat10desat10normdevdevvtest1',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/ext-blur9blur3blur1sharp9bright10dark10cont10decont10decont6sat10desat10normdevdevvtest1_train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/ext-blur9blur3blur1sharp9bright10dark10cont10decont10decont6sat10desat10normdevdevvtest1_val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})


facemasknosesextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_dataset = dataset_base.copy({
    'name': 'facemasknosesextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/ext-orft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/ext-orft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})


facemasknosesextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_dataset = dataset_base.copy({
    'name': 'facemasknosesextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/ext-g3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/ext-g3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})


facemasknosesada9_dataset = dataset_base.copy({
    'name': 'facemasknosesada9',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/ada9',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/ada9',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesada3_dataset = dataset_base.copy({
    'name': 'facemasknosesada3',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/ada3',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/ada3',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesblur9_dataset = dataset_base.copy({
    'name': 'facemasknosesblur9',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/blur9',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/blur9',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesblur1_dataset = dataset_base.copy({
    'name': 'facemasknosesblur1',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/blur1',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/blur1',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesblur3_dataset = dataset_base.copy({
    'name': 'facemasknosesblur3',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/blur3',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/blur3',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesbright10_dataset = dataset_base.copy({
    'name': 'facemasknosesbright10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/bright10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/bright10',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosescont10_dataset = dataset_base.copy({
    'name': 'facemasknosescont10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/cont10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/cont10',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesdark10_dataset = dataset_base.copy({
    'name': 'facemasknosesdark10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/dark10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/dark10',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesdecont10_dataset = dataset_base.copy({
    'name': 'facemasknosesdecont10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/decont10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/decont10',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesdecont6_dataset = dataset_base.copy({
    'name': 'facemasknosesdecont6',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/decont6',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/decont6',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesdesat10_dataset = dataset_base.copy({
    'name': 'facemasknosesdesat10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/desat10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/desat10',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesdev_dataset = dataset_base.copy({
    'name': 'facemasknosesdev',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/dev',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/dev',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesdevv_dataset = dataset_base.copy({
    'name': 'facemasknosesdevv',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/devv',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/devv',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesgray_dataset = dataset_base.copy({
    'name': 'facemasknosesgray',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/gray',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/gray',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesgray0_dataset = dataset_base.copy({
    'name': 'facemasknosesgray0',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/gray0',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/gray0',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknoseshsv_dataset = dataset_base.copy({
    'name': 'facemasknoseshsv',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/hsv',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/hsv',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesycc_dataset = dataset_base.copy({
    'name': 'facemasknosesycc',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/ycc',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/ycc',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknoseslab_dataset = dataset_base.copy({
    'name': 'facemasknoseslab',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/lab',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/lab',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknoseslab0_dataset = dataset_base.copy({
    'name': 'facemasknoseslab0',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/lab0',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/lab0',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesnorm_dataset = dataset_base.copy({
    'name': 'facemasknosesnorm',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/norm',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/norm',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosessat10_dataset = dataset_base.copy({
    'name': 'facemasknosessat10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/sat10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/sat10',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosessharp9_dataset = dataset_base.copy({
    'name': 'facemasknosessharp9',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/sharp9',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/sharp9',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosestest1_dataset = dataset_base.copy({
    'name': 'facemasknosestest1',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/test1',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/test1',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosestest2_dataset = dataset_base.copy({
    'name': 'facemasknosestest2',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/test2',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/test2',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesgray3blur3_dataset = dataset_base.copy({
    'name': 'facemasknosesgray3blur3',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/gray3blur3',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/gray3blur3',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesgray3blur1_dataset = dataset_base.copy({
    'name': 'facemasknosesgray3blur1',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/gray3blur1',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/gray3blur1',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesgray3ada9_dataset = dataset_base.copy({
    'name': 'facemasknosesgray3ada9',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/gray3ada9',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/gray3ada9',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesgray3ada3_dataset = dataset_base.copy({
    'name': 'facemasknosesgray3ada3',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/gray3ada3',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/gray3ada3',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesgray3bright10_dataset = dataset_base.copy({
    'name': 'facemasknosesgray3bright10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/gray3bright10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/gray3bright10',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesgray3dark10_dataset = dataset_base.copy({
    'name': 'facemasknosesgray3dark10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/gray3dark10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/gray3dark10',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesgray3cont10_dataset = dataset_base.copy({
    'name': 'facemasknosesgray3cont10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/gray3cont10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/gray3cont10',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesgray3decont10_dataset = dataset_base.copy({
    'name': 'facemasknosesgray3decont10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/gray3decont10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/gray3decont10',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesgray3desat10_dataset = dataset_base.copy({
    'name': 'facemasknosesgray3desat10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/gray3desat10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/gray3desat10',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesgray3dev_dataset = dataset_base.copy({
    'name': 'facemasknosesgray3dev',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/gray3dev',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/gray3dev',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesgray3devv_dataset = dataset_base.copy({
    'name': 'facemasknosesgray3devv',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/gray3devv',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/gray3devv',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesgray3norm_dataset = dataset_base.copy({
    'name': 'facemasknosesgray3norm',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/gray3norm',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/gray3norm',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesgray3sat10_dataset = dataset_base.copy({
    'name': 'facemasknosesgray3sat10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/gray3sat10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/gray3sat10',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})

facemasknosesgray3sharp9_dataset = dataset_base.copy({
    'name': 'facemasknosesgray3sharp9',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/train_via_project_5Dec2020_13h12m_coco.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images/gray3sharp9',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/val_via_project_23Dec2020_14h27m_coco.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images/gray3sharp9',
    'class_names': ('nose', 'mask', 'face'),
    'label_map': { 1: 1, 2: 2, 3: 3 }
})


maskfacesnewwork12falsefaceswork1nbnc_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork12falsefaceswork1nbnc',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork12falsefaceswork1nbnc/train/2020-12-14-00-28-06-810471_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork12falsefaceswork1nbnc/train/images',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork12falsefaceswork1nbnc/valid/2020-12-14-00-28-06-810471_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork12falsefaceswork1nbnc/valid/images',
    'class_names': ('mask', 'masknone'),
    'label_map': { 1: 1, 2: 2 }
})

maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb/train/2021-04-05-12-00-27-875589_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb/train/images',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb/valid/2021-04-05-12-00-27-875589_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb/valid/images',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nbg3_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nbg3',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb/train/ext-g3_2021-04-05-12-00-27-875589_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb/train/images',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb/valid/ext-g3_2021-04-05-12-00-27-875589_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb/valid/images',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork12falsefaceswork1nb_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork12falsefaceswork1nb',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork12falsefaceswork1nb/train/2020-12-09-18-50-40-452331_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork12falsefaceswork1nb/train/images',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork12falsefaceswork1nb/valid/2020-12-09-18-50-40-452331_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork12falsefaceswork1nb/valid/images',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1vk1exp1nbextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nbextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/ext-orft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/ext-orft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1vk1exp1nbextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nb',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/ext-g3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/ext-g3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})



maskfacesnewwork0312added1ffw1a1vk1exp1nb_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nb',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1vk1exp1nbada9_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nbada9',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images/ada9',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images/ada9',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1vk1exp1nbblur9_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nbblur9',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images/blur9',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images/blur9',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1vk1exp1nbbright10_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nbbright10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images/bright10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images/bright10',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1vk1exp1nbcont10_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nbcont10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images/cont10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images/cont10',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1vk1exp1nbdark10_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nbdark10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images/dark10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images/dark10',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1vk1exp1nbdecont10_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nbdecont10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images/decont10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images/decont10',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1vk1exp1nbdesat10_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nbdesat10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images/desat10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images/desat10',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1vk1exp1nbdev_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nbdev',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images/dev',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images/dev',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1vk1exp1nbdevv_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nbdevv',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images/devv',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images/devv',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1vk1exp1nbgray_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nbgray',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images/gray',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images/gray',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1vk1exp1nbhsv_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nbhsv',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images/hsv',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images/hsv',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1vk1exp1nblab_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nblab',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images/lab',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images/lab',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1vk1exp1nbnorm_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nbnorm',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images/norm',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images/norm',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1vk1exp1nbsat10_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nbsat10',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images/sat10',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images/sat10',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork0312added1ffw1a1vk1exp1nbsharp9_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork0312added1ffw1a1vk1exp1nbsharp9',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/2021-01-14-09-06-21-235911_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/train/images/sharp9',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/2021-01-14-09-06-21-235911_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork0312added1ffw1a1vk1exp1nb/valid/images/sharp9',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})

maskfacesnewwork12ffw1a1nbgray_dataset = dataset_base.copy({
    'name': 'maskfacesnewwork12ffw1a1nbgray',
    'train_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork12ffw1a1nbgray/train/2021-01-15-21-39-57-609992_coco_train.json',
    'train_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork12ffw1a1nbgray/train/images/gray3',
    'valid_info': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork12ffw1a1nbgray/valid/2021-01-15-21-39-57-609992_coco_valid.json',
    'valid_images': 'C:/Work/InfraredCamera/ThermalView/tests/train_models/dataset_outputs/maskfacesnewwork12ffw1a1nbgray/valid/images/gray3',
    'class_names': ('mask', 'maskchin', 'masknone', 'masknose'),
    'label_map': { 1: 1, 2: 2, 3: 3, 4: 4 }
})


# ----------------------- TRANSFORMS ----------------------- #

resnet_transform = Config({
    'channel_order': 'RGB',
    'normalize': True,
    'subtract_means': False,
    'to_float': False,
})

vgg_transform = Config({
    # Note that though vgg is traditionally BGR,
    # the channel order of vgg_reducedfc.pth is RGB.
    'channel_order': 'RGB',
    'normalize': False,
    'subtract_means': True,
    'to_float': False,
})

darknet_transform = Config({
    'channel_order': 'RGB',
    'normalize': False,
    'subtract_means': False,
    'to_float': True,
})





# ----------------------- BACKBONES ----------------------- #

backbone_base = Config({
    'name': 'Base Backbone',
    'path': 'path/to/pretrained/weights',
    'type': object,
    'args': tuple(),
    'transform': resnet_transform,

    'selected_layers': list(),
    'pred_scales': list(),
    'pred_aspect_ratios': list(),

    'use_pixel_scales': False,
    'preapply_sqrt': True,
    'use_square_anchors': False,
})

resnet101_backbone = backbone_base.copy({
    'name': 'ResNet101',
    'path': 'resnet101_reducedfc.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 23, 3],),
    'transform': resnet_transform,

    'selected_layers': list(range(2, 8)),
    'pred_scales': [[1]]*6,
    'pred_aspect_ratios': [ [[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]] ] * 6,
})

resnet101_gn_backbone = backbone_base.copy({
    'name': 'ResNet101_GN',
    'path': 'R-101-GN.pkl',
    'type': ResNetBackboneGN,
    'args': ([3, 4, 23, 3],),
    'transform': resnet_transform,

    'selected_layers': list(range(2, 8)),
    'pred_scales': [[1]]*6,
    'pred_aspect_ratios': [ [[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]] ] * 6,
})

resnet101_dcn_inter3_backbone = resnet101_backbone.copy({
    'name': 'ResNet101_DCN_Interval3',
    'args': ([3, 4, 23, 3], [0, 4, 23, 3], 3),
})

resnet50_backbone = resnet101_backbone.copy({
    'name': 'ResNet50',
    'path': 'resnet50-19c8e357.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 6, 3],),
    'transform': resnet_transform,
})

resnet50_dcnv2_backbone = resnet50_backbone.copy({
    'name': 'ResNet50_DCNv2',
    'args': ([3, 4, 6, 3], [0, 4, 6, 3]),
})

darknet53_backbone = backbone_base.copy({
    'name': 'DarkNet53',
    'path': 'darknet53.pth',
    'type': DarkNetBackbone,
    'args': ([1, 2, 8, 8, 4],),
    'transform': darknet_transform,

    'selected_layers': list(range(3, 9)),
    'pred_scales': [[3.5, 4.95], [3.6, 4.90], [3.3, 4.02], [2.7, 3.10], [2.1, 2.37], [1.8, 1.92]],
    'pred_aspect_ratios': [ [[1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n], [1]] for n in [3, 5, 5, 5, 3, 3] ],
})

vgg16_arch = [[64, 64],
              [ 'M', 128, 128],
              [ 'M', 256, 256, 256],
              [('M', {'kernel_size': 2, 'stride': 2, 'ceil_mode': True}), 512, 512, 512],
              [ 'M', 512, 512, 512],
              [('M',  {'kernel_size': 3, 'stride':  1, 'padding':  1}),
               (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}),
               (1024, {'kernel_size': 1})]]

vgg16_backbone = backbone_base.copy({
    'name': 'VGG16',
    'path': 'vgg16_reducedfc.pth',
    'type': VGGBackbone,
    'args': (vgg16_arch, [(256, 2), (128, 2), (128, 1), (128, 1)], [3]),
    'transform': vgg_transform,

    'selected_layers': [3] + list(range(5, 10)),
    'pred_scales': [[5, 4]]*6,
    'pred_aspect_ratios': [ [[1], [1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n]] for n in [3, 5, 5, 5, 3, 3] ],
})





# ----------------------- MASK BRANCH TYPES ----------------------- #

mask_type = Config({
    # Direct produces masks directly as the output of each pred module.
    # This is denoted as fc-mask in the paper.
    # Parameters: mask_size, use_gt_bboxes
    'direct': 0,

    # Lincomb produces coefficients as the output of each pred module then uses those coefficients
    # to linearly combine features from a prototype network to create image-sized masks.
    # Parameters:
    #   - masks_to_train (int): Since we're producing (near) full image masks, it'd take too much
    #                           vram to backprop on every single mask. Thus we select only a subset.
    #   - mask_proto_src (int): The input layer to the mask prototype generation network. This is an
    #                           index in backbone.layers. Use to use the image itself instead.
    #   - mask_proto_net (list<tuple>): A list of layers in the mask proto network with the last one
    #                                   being where the masks are taken from. Each conv layer is in
    #                                   the form (num_features, kernel_size, **kwdargs). An empty
    #                                   list means to use the source for prototype masks. If the
    #                                   kernel_size is negative, this creates a deconv layer instead.
    #                                   If the kernel_size is negative and the num_features is None,
    #                                   this creates a simple bilinear interpolation layer instead.
    #   - mask_proto_bias (bool): Whether to include an extra coefficient that corresponds to a proto
    #                             mask of all ones.
    #   - mask_proto_prototype_activation (func): The activation to apply to each prototype mask.
    #   - mask_proto_mask_activation (func): After summing the prototype masks with the predicted
    #                                        coeffs, what activation to apply to the final mask.
    #   - mask_proto_coeff_activation (func): The activation to apply to the mask coefficients.
    #   - mask_proto_crop (bool): If True, crop the mask with the predicted bbox during training.
    #   - mask_proto_crop_expand (float): If cropping, the percent to expand the cropping bbox by
    #                                     in each direction. This is to make the model less reliant
    #                                     on perfect bbox predictions.
    #   - mask_proto_loss (str [l1|disj]): If not None, apply an l1 or disjunctive regularization
    #                                      loss directly to the prototype masks.
    #   - mask_proto_binarize_downsampled_gt (bool): Binarize GT after dowsnampling during training?
    #   - mask_proto_normalize_mask_loss_by_sqrt_area (bool): Whether to normalize mask loss by sqrt(sum(gt))
    #   - mask_proto_reweight_mask_loss (bool): Reweight mask loss such that background is divided by
    #                                           #background and foreground is divided by #foreground.
    #   - mask_proto_grid_file (str): The path to the grid file to use with the next option.
    #                                 This should be a numpy.dump file with shape [numgrids, h, w]
    #                                 where h and w are w.r.t. the mask_proto_src convout.
    #   - mask_proto_use_grid (bool): Whether to add extra grid features to the proto_net input.
    #   - mask_proto_coeff_gate (bool): Add an extra set of sigmoided coefficients that is multiplied
    #                                   into the predicted coefficients in order to "gate" them.
    #   - mask_proto_prototypes_as_features (bool): For each prediction module, downsample the prototypes
    #                                 to the convout size of that module and supply the prototypes as input
    #                                 in addition to the already supplied backbone features.
    #   - mask_proto_prototypes_as_features_no_grad (bool): If the above is set, don't backprop gradients to
    #                                 to the prototypes from the network head.
    #   - mask_proto_remove_empty_masks (bool): Remove masks that are downsampled to 0 during loss calculations.
    #   - mask_proto_reweight_coeff (float): The coefficient to multiple the forground pixels with if reweighting.
    #   - mask_proto_coeff_diversity_loss (bool): Apply coefficient diversity loss on the coefficients so that the same
    #                                             instance has similar coefficients.
    #   - mask_proto_coeff_diversity_alpha (float): The weight to use for the coefficient diversity loss.
    #   - mask_proto_normalize_emulate_roi_pooling (bool): Normalize the mask loss to emulate roi pooling's affect on loss.
    #   - mask_proto_double_loss (bool): Whether to use the old loss in addition to any special new losses.
    #   - mask_proto_double_loss_alpha (float): The alpha to weight the above loss.
    #   - mask_proto_split_prototypes_by_head (bool): If true, this will give each prediction head its own prototypes.
    #   - mask_proto_crop_with_pred_box (bool): Whether to crop with the predicted box or the gt box.
    'lincomb': 1,
})





# ----------------------- ACTIVATION FUNCTIONS ----------------------- #

activation_func = Config({
    'tanh':    torch.tanh,
    'sigmoid': torch.sigmoid,
    'softmax': lambda x: torch.nn.functional.softmax(x, dim=-1),
    'relu':    lambda x: torch.nn.functional.relu(x, inplace=True),
    'none':    lambda x: x,
})





# ----------------------- FPN DEFAULTS ----------------------- #

fpn_base = Config({
    # The number of features to have in each FPN layer
    'num_features': 256,

    # The upsampling mode used
    'interpolation_mode': 'bilinear',

    # The number of extra layers to be produced by downsampling starting at P5
    'num_downsample': 1,

    # Whether to down sample with a 3x3 stride 2 conv layer instead of just a stride 2 selection
    'use_conv_downsample': False,

    # Whether to pad the pred layers with 1 on each side (I forgot to add this at the start)
    # This is just here for backwards compatibility
    'pad': True,

    # Whether to add relu to the downsampled layers.
    'relu_downsample_layers': False,

    # Whether to add relu to the regular layers
    'relu_pred_layers': True,
})





# ----------------------- CONFIG DEFAULTS ----------------------- #

coco_base_config = Config({
    'dataset': coco2014_dataset,
    'num_classes': 81, # This should include the background class

    'max_iter': 400000,

    # The maximum number of detections for evaluation
    'max_num_detections': 100,

    # dw' = momentum * dw - lr * (grad + decay * w)
    'lr': 1e-3,
    'momentum': 0.9,
    'decay': 5e-4,

    # For each lr step, what to multiply the lr with
    'gamma': 0.1,
    'lr_steps': (280000, 360000, 400000),

    # Initial learning rate to linearly warmup from (if until > 0)
    'lr_warmup_init': 1e-4,

    # If > 0 then increase the lr linearly from warmup_init to lr each iter for until iters
    'lr_warmup_until': 500,

    # The terms to scale the respective loss by
    'conf_alpha': 1,
    'bbox_alpha': 1.5,
    'mask_alpha': 0.4 / 256 * 140 * 140, # Some funky equation. Don't worry about it.

    # Eval.py sets this if you just want to run YOLACT as a detector
    'eval_mask_branch': True,

    # Top_k examples to consider for NMS
    'nms_top_k': 200,
    # Examples with confidence less than this are not considered by NMS
    'nms_conf_thresh': 0.05,
    # Boxes with IoU overlap greater than this threshold will be culled during NMS
    'nms_thresh': 0.5,

    # See mask_type for details.
    'mask_type': mask_type.direct,
    'mask_size': 16,
    'masks_to_train': 100,
    'mask_proto_src': None,
    'mask_proto_net': [(256, 3, {}), (256, 3, {})],
    'mask_proto_bias': False,
    'mask_proto_prototype_activation': activation_func.relu,
    'mask_proto_mask_activation': activation_func.sigmoid,
    'mask_proto_coeff_activation': activation_func.tanh,
    'mask_proto_crop': True,
    'mask_proto_crop_expand': 0,
    'mask_proto_loss': None,
    'mask_proto_binarize_downsampled_gt': True,
    'mask_proto_normalize_mask_loss_by_sqrt_area': False,
    'mask_proto_reweight_mask_loss': False,
    'mask_proto_grid_file': 'data/grid.npy',
    'mask_proto_use_grid':  False,
    'mask_proto_coeff_gate': False,
    'mask_proto_prototypes_as_features': False,
    'mask_proto_prototypes_as_features_no_grad': False,
    'mask_proto_remove_empty_masks': False,
    'mask_proto_reweight_coeff': 1,
    'mask_proto_coeff_diversity_loss': False,
    'mask_proto_coeff_diversity_alpha': 1,
    'mask_proto_normalize_emulate_roi_pooling': False,
    'mask_proto_double_loss': False,
    'mask_proto_double_loss_alpha': 1,
    'mask_proto_split_prototypes_by_head': False,
    'mask_proto_crop_with_pred_box': False,

    # SSD data augmentation parameters
    # Randomize hue, vibrance, etc.
    'augment_photometric_distort': True,
    # Have a chance to scale down the image and pad (to emulate smaller detections)
    'augment_expand': True,
    # Potentialy sample a random crop from the image and put it in a random place
    'augment_random_sample_crop': True,
    # Mirror the image with a probability of 1/2
    'augment_random_mirror': True,
    # Flip the image vertically with a probability of 1/2
    'augment_random_flip': False,
    # With uniform probability, rotate the image [0,90,180,270] degrees
    'augment_random_rot90': False,

    # Discard detections with width and height smaller than this (in absolute width and height)
    'discard_box_width': 4 / 550,
    'discard_box_height': 4 / 550,

    # If using batchnorm anywhere in the backbone, freeze the batchnorm layer during training.
    # Note: any additional batch norm layers after the backbone will not be frozen.
    'freeze_bn': False,

    # Set this to a config object if you want an FPN (inherit from fpn_base). See fpn_base for details.
    'fpn': None,

    # Use the same weights for each network head
    'share_prediction_module': False,

    # For hard negative mining, instead of using the negatives that are leastl confidently background,
    # use negatives that are most confidently not background.
    'ohem_use_most_confident': False,

    # Use focal loss as described in https://arxiv.org/pdf/1708.02002.pdf instead of OHEM
    'use_focal_loss': False,
    'focal_loss_alpha': 0.25,
    'focal_loss_gamma': 2,
    
    # The initial bias toward forground objects, as specified in the focal loss paper
    'focal_loss_init_pi': 0.01,

    # Keeps track of the average number of examples for each class, and weights the loss for that class accordingly.
    'use_class_balanced_conf': False,

    # Whether to use sigmoid focal loss instead of softmax, all else being the same.
    'use_sigmoid_focal_loss': False,

    # Use class[0] to be the objectness score and class[1:] to be the softmax predicted class.
    # Note: at the moment this is only implemented if use_focal_loss is on.
    'use_objectness_score': False,

    # Adds a global pool + fc layer to the smallest selected layer that predicts the existence of each of the 80 classes.
    # This branch is only evaluated during training time and is just there for multitask learning.
    'use_class_existence_loss': False,
    'class_existence_alpha': 1,

    # Adds a 1x1 convolution directly to the biggest selected layer that predicts a semantic segmentations for each of the 80 classes.
    # This branch is only evaluated during training time and is just there for multitask learning.
    'use_semantic_segmentation_loss': False,
    'semantic_segmentation_alpha': 1,

    # Adds another branch to the netwok to predict Mask IoU.
    'use_mask_scoring': False,
    'mask_scoring_alpha': 1,

    # Match gt boxes using the Box2Pix change metric instead of the standard IoU metric.
    # Note that the threshold you set for iou_threshold should be negative with this setting on.
    'use_change_matching': False,

    # Uses the same network format as mask_proto_net, except this time it's for adding extra head layers before the final
    # prediction in prediction modules. If this is none, no extra layers will be added.
    'extra_head_net': None,

    # What params should the final head layers have (the ones that predict box, confidence, and mask coeffs)
    'head_layer_params': {'kernel_size': 3, 'padding': 1},

    # Add extra layers between the backbone and the network heads
    # The order is (bbox, conf, mask)
    'extra_layers': (0, 0, 0),

    # During training, to match detections with gt, first compute the maximum gt IoU for each prior.
    # Then, any of those priors whose maximum overlap is over the positive threshold, mark as positive.
    # For any priors whose maximum is less than the negative iou threshold, mark them as negative.
    # The rest are neutral and not used in calculating the loss.
    'positive_iou_threshold': 0.5,
    'negative_iou_threshold': 0.5,

    # When using ohem, the ratio between positives and negatives (3 means 3 negatives to 1 positive)
    'ohem_negpos_ratio': 3,

    # If less than 1, anchors treated as a negative that have a crowd iou over this threshold with
    # the crowd boxes will be treated as a neutral.
    'crowd_iou_threshold': 1,

    # This is filled in at runtime by Yolact's __init__, so don't touch it
    'mask_dim': None,

    # Input image size.
    'max_size': 300,
    
    # Whether or not to do post processing on the cpu at test time
    'force_cpu_nms': True,

    # Whether to use mask coefficient cosine similarity nms instead of bbox iou nms
    'use_coeff_nms': False,

    # Whether or not to have a separate branch whose sole purpose is to act as the coefficients for coeff_diversity_loss
    # Remember to turn on coeff_diversity_loss, or these extra coefficients won't do anything!
    # To see their effect, also remember to turn on use_coeff_nms.
    'use_instance_coeff': False,
    'num_instance_coeffs': 64,

    # Whether or not to tie the mask loss / box loss to 0
    'train_masks': True,
    'train_boxes': True,
    # If enabled, the gt masks will be cropped using the gt bboxes instead of the predicted ones.
    # This speeds up training time considerably but results in much worse mAP at test time.
    'use_gt_bboxes': False,

    # Whether or not to preserve aspect ratio when resizing the image.
    # If True, this will resize all images to be max_size^2 pixels in area while keeping aspect ratio.
    # If False, all images are resized to max_size x max_size
    'preserve_aspect_ratio': False,

    # Whether or not to use the prediction module (c) from DSSD
    'use_prediction_module': False,

    # Whether or not to use the predicted coordinate scheme from Yolo v2
    'use_yolo_regressors': False,
    
    # For training, bboxes are considered "positive" if their anchors have a 0.5 IoU overlap
    # or greater with a ground truth box. If this is true, instead of using the anchor boxes
    # for this IoU computation, the matching function will use the predicted bbox coordinates.
    # Don't turn this on if you're not using yolo regressors!
    'use_prediction_matching': False,

    # A list of settings to apply after the specified iteration. Each element of the list should look like
    # (iteration, config_dict) where config_dict is a dictionary you'd pass into a config object's init.
    'delayed_settings': [],

    # Use command-line arguments to set this.
    'no_jit': False,

    'backbone': None,
    'name': 'base_config',

    # Fast Mask Re-scoring Network
    # Inspried by Mask Scoring R-CNN (https://arxiv.org/abs/1903.00241)
    # Do not crop out the mask with bbox but slide a convnet on the image-size mask,
    # then use global pooling to get the final mask score
    'use_maskiou': False,
    
    # Archecture for the mask iou network. A (num_classes-1, 1, {}) layer is appended to the end.
    'maskiou_net': [],

    # Discard predicted masks whose area is less than this
    'discard_mask_area': -1,

    'maskiou_alpha': 1.0,
    'rescore_mask': False,
    'rescore_bbox': False,
    'maskious_to_train': -1,
})





# ----------------------- YOLACT v1.0 CONFIGS ----------------------- #

yolact_base_config = coco_base_config.copy({
    'name': 'yolact_base',

    # Dataset stuff
    'dataset': coco2017_dataset,
    'num_classes': len(coco2017_dataset.class_names) + 1,

    # Image Size
    'max_size': 550,
    
    # Training params
    'lr_steps': (280000, 600000, 700000, 750000),
    'max_iter': 800000,
    
    # Backbone Settings
    'backbone': resnet101_backbone.copy({
        'selected_layers': list(range(1, 4)),
        'use_pixel_scales': True,
        'preapply_sqrt': False,
        'use_square_anchors': True, # This is for backward compatability with a bug

        'pred_aspect_ratios': [ [[1, 1/2, 2]] ]*5,
        'pred_scales': [[24], [48], [96], [192], [384]],
    }),

    # FPN Settings
    'fpn': fpn_base.copy({
        'use_conv_downsample': True,
        'num_downsample': 2,
    }),

    # Mask Settings
    'mask_type': mask_type.lincomb,
    'mask_alpha': 6.125,
    'mask_proto_src': 0,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})],
    'mask_proto_normalize_emulate_roi_pooling': True,

    # Other stuff
    'share_prediction_module': True,
    'extra_head_net': [(256, 3, {'padding': 1})],

    'positive_iou_threshold': 0.5,
    'negative_iou_threshold': 0.4,

    'crowd_iou_threshold': 0.7,

    'use_semantic_segmentation_loss': True,
})

yolact_im400_config = yolact_base_config.copy({
    'name': 'yolact_im400',

    'max_size': 400,
    'backbone': yolact_base_config.backbone.copy({
        'pred_scales': [[int(x[0] / yolact_base_config.max_size * 400)] for x in yolact_base_config.backbone.pred_scales],
    }),
})

yolact_im700_config = yolact_base_config.copy({
    'name': 'yolact_im700',

    'masks_to_train': 300,
    'max_size': 700,
    'backbone': yolact_base_config.backbone.copy({
        'pred_scales': [[int(x[0] / yolact_base_config.max_size * 700)] for x in yolact_base_config.backbone.pred_scales],
    }),
})

yolact_darknet53_config = yolact_base_config.copy({
    'name': 'yolact_darknet53',

    'backbone': darknet53_backbone.copy({
        'selected_layers': list(range(2, 5)),
        
        'pred_scales': yolact_base_config.backbone.pred_scales,
        'pred_aspect_ratios': yolact_base_config.backbone.pred_aspect_ratios,
        'use_pixel_scales': True,
        'preapply_sqrt': False,
        'use_square_anchors': True, # This is for backward compatability with a bug
    }),
})

yolact_resnet50_config = yolact_base_config.copy({
    'name': 'yolact_resnet50',

    'backbone': resnet50_backbone.copy({
        'selected_layers': list(range(1, 4)),
        
        'pred_scales': yolact_base_config.backbone.pred_scales,
        'pred_aspect_ratios': yolact_base_config.backbone.pred_aspect_ratios,
        'use_pixel_scales': True,
        'preapply_sqrt': False,
        'use_square_anchors': True, # This is for backward compatability with a bug
    }),
})


yolact_resnet50_pascal_config = yolact_resnet50_config.copy({
    'name': None, # Will default to yolact_resnet50_pascal
    
    # Dataset stuff
    'dataset': pascal_sbd_dataset,
    'num_classes': len(pascal_sbd_dataset.class_names) + 1,

    'max_iter': 120000,
    'lr_steps': (60000, 100000),
    
    'backbone': yolact_resnet50_config.backbone.copy({
        'pred_scales': [[32], [64], [128], [256], [512]],
        'use_square_anchors': False,
    })
})

yolact_maskfaces_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfaces',
    # Dataset stuff
    'dataset': mask_faces_dataset,
    'num_classes': len(mask_faces_dataset.class_names) + 1,

    # Image Size
    'max_size': 512,
})

yolact_heads_config = yolact_resnet50_config.copy({
    'name': 'yolact_heads',
    # Dataset stuff
    'dataset': heads_dataset,
    'num_classes': len(heads_dataset.class_names) + 1,

    # Image Size
    'max_size': 512,
})

yolact_headsgray_config = yolact_resnet50_config.copy({
    'name': 'yolact_headsgray',
    # Dataset stuff
    'dataset': headsgray_dataset,
    'num_classes': len(headsgray_dataset.class_names) + 1,

    # Image Size
    'max_size': 512,
})

yolact_heads416_config = yolact_resnet50_config.copy({
    'name': 'yolact_heads416',
    # Dataset stuff
    'dataset': heads_dataset,
    'num_classes': len(heads_dataset.class_names) + 1,

    # Image Size
    'max_size': 416,
})

yolact_headsnew_config = yolact_resnet50_config.copy({
    'name': 'yolact_headsnew',
    # Dataset stuff
    'dataset': headsnew_dataset,
    'num_classes': len(headsnew_dataset.class_names) + 1,

    # Image Size
    'max_size': 512,
})

yolact_headsnewgray_config = yolact_resnet50_config.copy({
    'name': 'yolact_headsnewgray',
    # Dataset stuff
    'dataset': headsnewgray_dataset,
    'num_classes': len(headsnewgray_dataset.class_names) + 1,

    # Image Size
    'max_size': 512,
})

yolact_headsnoses_config = yolact_resnet50_config.copy({
    'name': 'yolact_headsnoses',
    # Dataset stuff
    'dataset': headsnoses_dataset,
    'num_classes': len(headsnoses_dataset.class_names) + 1,

    # Image Size
    'max_size': 512,
})

yolact_maskfacesnew_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnew',
    'dataset': maskfacesnew_dataset,
    'num_classes': len(maskfacesnew_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnew2_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnew2',
    'dataset': maskfacesnew2_dataset,
    'num_classes': len(maskfacesnew2_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnew03_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnew03',
    'dataset': maskfacesnew03_dataset,
    'num_classes': len(maskfacesnew03_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312mask_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312mask',
    'dataset': maskfacesnewwork0312mask_dataset,
    'num_classes': len(maskfacesnewwork0312mask_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1nb_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1nb',
    'dataset': maskfacesnewwork0312added1nb_dataset,
    'num_classes': len(maskfacesnewwork0312added1nb_dataset.class_names) + 1,
    'max_size': 512
})

yolact_noses_config = yolact_resnet50_config.copy({
    'name': 'yolact_noses',
    'dataset': noses_dataset,
    'num_classes': len(noses_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknoses_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknoses',
    'dataset': facemasknoses_dataset,
    'num_classes': len(facemasknoses_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesextblur9blur3blur1sharp9bright10dark10cont10decont10decont6sat10desat10normdevdevvtest1_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesextblur9blur3blur1sharp9bright10dark10cont10decont10decont6sat10desat10normdevdevvtest1',
    'dataset': facemasknosesextblur9blur3blur1sharp9bright10dark10cont10decont10decont6sat10desat10normdevdevvtest1_dataset,
    'num_classes': len(facemasknosesextblur9blur3blur1sharp9bright10dark10cont10decont10decont6sat10desat10normdevdevvtest1_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf',
    'dataset': facemasknosesextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_dataset,
    'num_classes': len(facemasknosesextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvfnobn_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvfnobn',
    'dataset': facemasknosesextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_dataset,
    'num_classes': len(facemasknosesextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf640_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf640',
    'dataset': facemasknosesextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_dataset,
    'num_classes': len(facemasknosesextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_dataset.class_names) + 1,
    'max_size': 640
})

yolact_facemasknosesextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv',
    'dataset': facemasknosesextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_dataset,
    'num_classes': len(facemasknosesextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvvnobn_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvvnobn',
    'dataset': facemasknosesextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_dataset,
    'num_classes': len(facemasknosesextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv640_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv640',
    'dataset': facemasknosesextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_dataset,
    'num_classes': len(facemasknosesextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_dataset.class_names) + 1,
    'max_size': 640
})

yolact_facemasknosesada9_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesada9',
    'dataset': facemasknosesada9_dataset,
    'num_classes': len(facemasknosesada9_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesada3_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesada3',
    'dataset': facemasknosesada3_dataset,
    'num_classes': len(facemasknosesada3_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesblur9_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesblur9',
    'dataset': facemasknosesblur9_dataset,
    'num_classes': len(facemasknosesblur9_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesblur3_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesblur3',
    'dataset': facemasknosesblur3_dataset,
    'num_classes': len(facemasknosesblur3_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesblur1_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesblur1',
    'dataset': facemasknosesblur1_dataset,
    'num_classes': len(facemasknosesblur1_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesbright10_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesbright10',
    'dataset': facemasknosesbright10_dataset,
    'num_classes': len(facemasknosesbright10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosescont10_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosescont10',
    'dataset': facemasknosescont10_dataset,
    'num_classes': len(facemasknosescont10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesdark10_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesdark10',
    'dataset': facemasknosesdark10_dataset,
    'num_classes': len(facemasknosesdark10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesdecont10_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesdecont10',
    'dataset': facemasknosesdecont10_dataset,
    'num_classes': len(facemasknosesdecont10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesdecont6_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesdecont6',
    'dataset': facemasknosesdecont6_dataset,
    'num_classes': len(facemasknosesdecont6_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesdesat10_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesdesat10',
    'dataset': facemasknosesdesat10_dataset,
    'num_classes': len(facemasknosesdesat10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesdev_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesdev',
    'dataset': facemasknosesdev_dataset,
    'num_classes': len(facemasknosesdev_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesdevv_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesdevv',
    'dataset': facemasknosesdevv_dataset,
    'num_classes': len(facemasknosesdevv_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesgray_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesgray',
    'dataset': facemasknosesgray_dataset,
    'num_classes': len(facemasknosesgray_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesgray640_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesgray640',
    'dataset': facemasknosesgray_dataset,
    'num_classes': len(facemasknosesgray_dataset.class_names) + 1,
    'max_size': 640
})

yolact_facemasknosesgray0_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesgray0',
    'dataset': facemasknosesgray0_dataset,
    'num_classes': len(facemasknosesgray0_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknoseshsv_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknoseshsv',
    'dataset': facemasknoseshsv_dataset,
    'num_classes': len(facemasknoseshsv_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesycc_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesycc',
    'dataset': facemasknosesycc_dataset,
    'num_classes': len(facemasknosesycc_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknoseslab_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknoseslab',
    'dataset': facemasknoseslab_dataset,
    'num_classes': len(facemasknoseslab_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknoseslab0_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknoseslab0',
    'dataset': facemasknoseslab0_dataset,
    'num_classes': len(facemasknoseslab0_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesnorm_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesnorm',
    'dataset': facemasknosesnorm_dataset,
    'num_classes': len(facemasknosesnorm_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosessat10_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosessat10',
    'dataset': facemasknosessat10_dataset,
    'num_classes': len(facemasknosessat10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosessharp9_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosessharp9',
    'dataset': facemasknosessharp9_dataset,
    'num_classes': len(facemasknosessharp9_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosestest1_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosestest1',
    'dataset': facemasknosestest1_dataset,
    'num_classes': len(facemasknosestest1_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosestest2_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosestest2',
    'dataset': facemasknosestest2_dataset,
    'num_classes': len(facemasknosestest2_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesgray3blur3_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesgray3blur3',
    'dataset': facemasknosesgray3blur3_dataset,
    'num_classes': len(facemasknosesgray3blur3_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesgray3blur1_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesgray3blur1',
    'dataset': facemasknosesgray3blur1_dataset,
    'num_classes': len(facemasknosesgray3blur1_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesgray3ada9_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesgray3ada9',
    'dataset': facemasknosesgray3ada9_dataset,
    'num_classes': len(facemasknosesgray3ada9_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesgray3ada3_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesgray3ada3',
    'dataset': facemasknosesgray3ada3_dataset,
    'num_classes': len(facemasknosesgray3ada3_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesgray3bright10_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesgray3bright10',
    'dataset': facemasknosesgray3bright10_dataset,
    'num_classes': len(facemasknosesgray3bright10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesgray3dark10_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesgray3dark10',
    'dataset': facemasknosesgray3dark10_dataset,
    'num_classes': len(facemasknosesgray3dark10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesgray3cont10_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesgray3cont10',
    'dataset': facemasknosesgray3cont10_dataset,
    'num_classes': len(facemasknosesgray3cont10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesgray3decont10_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesgray3decont10',
    'dataset': facemasknosesgray3decont10_dataset,
    'num_classes': len(facemasknosesgray3decont10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesgray3desat10_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesgray3desat10',
    'dataset': facemasknosesgray3desat10_dataset,
    'num_classes': len(facemasknosesgray3desat10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesgray3dev_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesgray3dev',
    'dataset': facemasknosesgray3dev_dataset,
    'num_classes': len(facemasknosesgray3dev_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesgray3devv_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesgray3devv',
    'dataset': facemasknosesgray3devv_dataset,
    'num_classes': len(facemasknosesgray3devv_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesgray3norm_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesgray3norm',
    'dataset': facemasknosesgray3norm_dataset,
    'num_classes': len(facemasknosesgray3norm_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesgray3sat10_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesgray3sat10',
    'dataset': facemasknosesgray3sat10_dataset,
    'num_classes': len(facemasknosesgray3sat10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_facemasknosesgray3sharp9_config = yolact_resnet50_config.copy({
    'name': 'yolact_facemasknosesgray3sharp9',
    'dataset': facemasknosesgray3sharp9_dataset,
    'num_classes': len(facemasknosesgray3sharp9_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork12falsefaceswork1nbnc_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork12falsefaceswork1nbnc',
    'dataset': maskfacesnewwork12falsefaceswork1nbnc_dataset,
    'num_classes': len(maskfacesnewwork12falsefaceswork1nbnc_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork12falsefaceswork1nb_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork12falsefaceswork1nb',
    'dataset': maskfacesnewwork12falsefaceswork1nb_dataset,
    'num_classes': len(maskfacesnewwork12falsefaceswork1nb_dataset.class_names) + 1,
    'max_size': 512
})


yolact_maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb',
    'dataset': maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb640_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb640',
    'dataset': maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb_dataset.class_names) + 1,
    'max_size': 640
})

yolact_maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nbg3_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nbg3',
    'dataset': maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nbg3_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nbg3_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nbg3640_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nbg3640',
    'dataset': maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nbg3_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nbg3_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf640_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf640',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbextorft1t1fg3g3fbl3bl3fsh9sh9fbr10br10fdr10dr10fcn10cn10fdc10dc10fnrnrfdvdvfvvvvf_dataset.class_names) + 1,
    'max_size': 640
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv640_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv640',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbextg3g3fg3bl3g3fbl3g3sh9g3fsh9g3br10g3fbr10g3dr10g3fdr10g3cn10g3fcn10g3dc10g3fdc10g3nrg3fnrg3dvg3fdvg3vvg3fvv_dataset.class_names) + 1,
    'max_size': 640
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nb_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nb',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nb_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nb_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbada9_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbada9',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbada9_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbada9_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbblur9_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbblur9',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbblur9_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbblur9_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbbright10_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbbright10',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbbright10_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbbright10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbcont10_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbcont10',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbcont10_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbcont10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbdark10_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbdark10',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbdark10_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbdark10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbdecont10_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbdecont10',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbdecont10_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbdecont10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbdesat10_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbdesat10',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbdesat10_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbdesat10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbdev_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbdev',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbdev_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbdev_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbdevv_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbdevv',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbdevv_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbdevv_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbgray_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbgray',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbgray_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbgray_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbgray640_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbgray640',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbgray_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbgray_dataset.class_names) + 1,
    'max_size': 640
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbhsv_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbhsv',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbhsv_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbhsv_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nblab_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nblab',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nblab_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nblab_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbnorm_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbnorm',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbnorm_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbnorm_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbsat10_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbsat10',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbsat10_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbsat10_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbsharp9_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbsharp9',
    'dataset': maskfacesnewwork0312added1ffw1a1vk1exp1nbsharp9_dataset,
    'num_classes': len(maskfacesnewwork0312added1ffw1a1vk1exp1nbsharp9_dataset.class_names) + 1,
    'max_size': 512
})

yolact_maskfacesnewwork12ffw1a1nbgray_config = yolact_resnet50_config.copy({
    'name': 'yolact_maskfacesnewwork12ffw1a1nbgray',
    'dataset': maskfacesnewwork12ffw1a1nbgray_dataset,
    'num_classes': len(maskfacesnewwork12ffw1a1nbgray_dataset.class_names) + 1,
    'max_size': 512
})

# ----------------------- YOLACT++ CONFIGS ----------------------- #

yolact_plus_base_config = yolact_base_config.copy({
    'name': 'yolact_plus_base',

    'backbone': resnet101_dcn_inter3_backbone.copy({
        'selected_layers': list(range(1, 4)),
        
        'pred_aspect_ratios': [ [[1, 1/2, 2]] ]*5,
        'pred_scales': [[i * 2 ** (j / 3.0) for j in range(3)] for i in [24, 48, 96, 192, 384]],
        'use_pixel_scales': True,
        'preapply_sqrt': False,
        'use_square_anchors': False,
    }),

    'use_maskiou': True,
    'maskiou_net': [(8, 3, {'stride': 2}), (16, 3, {'stride': 2}), (32, 3, {'stride': 2}), (64, 3, {'stride': 2}), (128, 3, {'stride': 2})],
    'maskiou_alpha': 25,
    'rescore_bbox': False,
    'rescore_mask': True,

    'discard_mask_area': 5*5,
})

yolact_plus_resnet50_config = yolact_plus_base_config.copy({
    'name': 'yolact_plus_resnet50',

    'backbone': resnet50_dcnv2_backbone.copy({
        'selected_layers': list(range(1, 4)),
        
        'pred_aspect_ratios': [ [[1, 1/2, 2]] ]*5,
        'pred_scales': [[i * 2 ** (j / 3.0) for j in range(3)] for i in [24, 48, 96, 192, 384]],
        'use_pixel_scales': True,
        'preapply_sqrt': False,
        'use_square_anchors': False,
    }),
})


# Default config
cfg = yolact_base_config.copy()

def set_cfg(config_name:str):
    """ Sets the active config. Works even if cfg is already imported! """
    global cfg

    # Note this is not just an eval because I'm lazy, but also because it can
    # be used like ssd300_config.copy({'max_size': 400}) for extreme fine-tuning
    cfg.replace(eval(config_name))

    if cfg.name is None:
        cfg.name = config_name.split('_config')[0]

def set_dataset(dataset_name:str):
    """ Sets the dataset of the current config. """
    cfg.dataset = eval(dataset_name)
    
