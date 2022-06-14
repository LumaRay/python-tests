import json
import os
import pathlib
import random
import shutil
from datetime import datetime

pathToScriptFolder = str(pathlib.Path().absolute().as_posix())

#DATASET_NAME = 'maskfacesnew03'
#DATASET_NAME = 'maskfacesnewwork0312mask'
#DATASET_NAME = 'maskfacesnewwork0312added1nb'
#DATASET_NAME = 'maskfacesnewwork12falsefaceswork1nb'
#DATASET_NAME = 'maskfacesnewwork12falsefaceswork1nbnc'
#DATASET_NAME = 'maskfacesnewwork12falsefaceswork1added1nb'
#DATASET_NAME = 'maskfacesnewwork0312added1ffw1a1vk1exp1'

DATASET_NAMES = [
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
    'v2_source_2020_11_17',
    'v2_source_2020_11_19',
    'v2_source_2020_11_25_1000ms',
    'v2_source_2020_11_25_1000ms5max',
    'v2_source_2020_11_25_200ms',
    'v2_source_2020_11_25_3000ms',
    'v2_source_2020_11_25_3000ms5max',
    'v2_source_2020_11_25_300ms',
    'v2_source_2020_11_25_noms',
    'v2_vk_sources2_sorted1',  # not work
]

CLASS_NAMES = ['mask', 'maskchin', 'masknone', 'masknose']  #  NO BACK !!!
CLASSES_MAP = [0, 1, 2, 3]  # All classes
# CLASSES_MAP = [0, 2, 2, 2]  # mask/masknone classes

# CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']
# CLASSES_MAP = [0, 1, 2, 3, 4]  # All classes

# CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']
# CLASSES_MAP = [0, 0, 0, 0, 0]  # All classes

OUTPUT_DATASET_NAME = 'maskfacesnewwork0312added1ffw1a1exp1v12vk1v1vk2v2p1nb'
# OUTPUT_DATASET_NAME = 'maskfacesnewwork0312added1ffw1a1vk1exp1nb'
# OUTPUT_DATASET_NAME = 'maskfacesnewwork12ffw1a1nbgray'
# OUTPUT_DATASET_NAME = 'maskfacesnewwork12ffw1a1v12nbgray'
# OUTPUT_DATASET_NAME = 'headsnew'

VALID_RATIO = 0.2

MAX_IMAGE_SIZE = 512
#MAX_IMAGE_SIZE = 416

# EXPORT_YOLO = False
EXPORT_YOLO = True

CUT_EVEN_CLASS_DATASETS = False
# CUT_EVEN_CLASS_DATASETS = True

dataset_sources_path = pathToScriptFolder + '/../dataset_sources'
#classes_source_path = dataset_sources_path + '/' + DATASET_NAME

#if not os.path.exists(classes_source_path):
#    os.makedirs(classes_source_path)

parsed_data_path = pathToScriptFolder + '/../parsed_data'
parsed_data_last_path_list = []

dataset_outputs_path = pathToScriptFolder + '/../dataset_outputs'
dataset_output_path = dataset_outputs_path + '/' + OUTPUT_DATASET_NAME
dataset_output_train_path = dataset_output_path + '/train'
dataset_output_train_images_path = dataset_output_train_path + '/images'
dataset_output_valid_path = dataset_output_path + '/valid'
dataset_output_valid_images_path = dataset_output_valid_path + '/images'

if not os.path.exists(dataset_output_train_images_path):
    os.makedirs(dataset_output_train_images_path)
if not os.path.exists(dataset_output_valid_images_path):
    os.makedirs(dataset_output_valid_images_path)

#parsed_data_path = pathToScriptFolder + '/parsed_data/'
#parsed_data_path = pathToScriptFolder + '/parsed_data/' + 'parsed_data_work0312/'
#parsed_data_path = pathToScriptFolder + '/parsed_data/' + 'parsed_data_work12_false_faces_work1/'
#parsed_data_path = pathToScriptFolder + '/../parsed_data/' + 'parsed_data_work12_false_faces_work1_added1/'
#parsed_data_path = pathToScriptFolder + '/../parsed_data/' + 'parsed_data_work0312_false_faces_work1_added1_vk_sources1exp1/'
#objects_source_path = parsed_data_path + 'objects'
#images_source_path = parsed_data_path + 'frames'

#heads_source_path = pathToScriptFolder + '/parsed_data/heads'
#people_source_path = pathToScriptFolder + '/parsed_data/people'
#frames_source_path = pathToScriptFolder + '/parsed_data/frames'

#lst = os.walk(pathToScriptFolder)

#class_pathes = [ f.path for f in os.scandir(classes_source_path) if f.is_dir() ]

dict_images = {}

dict_categories = {}

#annotations = []

#image_cnt = 0

#category_id = 0

#annotation_id = 1

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

class_images_counts = {}
for class_idx, class_name in enumerate(CLASS_NAMES):
    class_images_counts[class_name] = 0
    for dataset_name in DATASET_NAMES:
        class_path = dataset_sources_path + '/' + dataset_name + '/' + class_name
        files = os.listdir(class_path)
        for fidx, file in enumerate(files):
            if file.endswith(".jpg"):
                class_images_counts[class_name] += 1

total_images_count = 0
min_class_images_count = 9999999
for class_idx, (class_name, class_images_count) in enumerate(class_images_counts.items()):
    min_class_images_count = min(min_class_images_count, class_images_count)
    total_images_count += class_images_count

'''class_images_cut_ratios = {}
for class_idx, (class_name, class_images_count) in enumerate(class_images_counts):
    class_images_cut_ratios[class_name] = min_class_images_count / class_images_count'''

#for class_path in class_pathes:
for class_idx, class_name in enumerate(CLASS_NAMES):
    #class_name = os.path.basename(os.path.normpath(class_path))
    print("Parsing class " + class_name)
    for dataset_name in DATASET_NAMES:
        print("Parsing dataset " + dataset_name)
        class_path = dataset_sources_path + '/' + dataset_name + '/' + class_name
        new_class_idx = CLASSES_MAP[class_idx]
        # if new_class_idx != class_idx:
        new_class_name = CLASS_NAMES[new_class_idx]
        print("Mapping class {} to {}".format(class_name, new_class_name))
        class_name = new_class_name
        # class_idx = new_class_idx
        if dict_categories.get(class_name) is None:
            category_id = len(dict_categories) + 1
            dict_categories[class_name] = category_id
        else:
            category_id = dict_categories[class_name]
        files = os.listdir(class_path)
        for fidx, file in enumerate(files):
            if file.endswith(".jpg"):
                if CUT_EVEN_CLASS_DATASETS:
                    if random.uniform(0, 1) > (min_class_images_count / class_images_counts[class_name]):
                        continue
                file_name, file_ext = file.split('.')
                #print("Parsing file " + str(fidx) + " of " + str(len(files)))
                frame_timestamp, score, timestamp = file_name.split('_')
                object_path = findParsedFile(file_name + '.txt')
                print("Parsing class=", class_name, "dataset=", dataset_name, "file", fidx, "of", len(files), file_name, "object_path=", object_path)
                #with open(objects_source_path + '/' + file_name + '.txt') as json_file:
                if object_path is None:
                    input("Press Enter to continue...")
                    continue
                with open(object_path) as json_file:
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
                    if dict_images.get(frame_ref) is None:
                        dict_images[frame_ref] = {}
                        dict_images[frame_ref]["id"] = -1 # len(dict_images)
                        dict_images[frame_ref]["width"] = frame_width
                        dict_images[frame_ref]["height"] = frame_height
                        dict_images[frame_ref]["file_name"] = frame_ref + ".jpg"
                        dict_images[frame_ref]["license"] = 0
                        dict_images[frame_ref]["date_captured"] = ""
                        dict_images[frame_ref]["annotations"] = []
                    annotations = dict_images[frame_ref]["annotations"]
                    #image_id = dict_images[frame_ref]["id"]
                    annotation = {}
                    annotation["area"] = contour_area
                    annotation["bbox"] = [bbox_x, bbox_y, bbox_w, bbox_h]
                    annotation["iscrowd"] = 0
                    annotation["id"] = -1 # annotation_id
                    annotation["image_id"] = -1 # image_id
                    annotation["category_id"] = category_id
                    annotation["segmentation"] = []
                    segmentation = []
                    for contour_point in contour:
                        x = contour_point['x']
                        segmentation.append(x)
                        y = contour_point['y']
                        segmentation.append(y)
                    annotation["segmentation"].append(segmentation)
                    annotations.append(annotation)
                    # annotation_id += 1


n_images = len(dict_images)

valid_idxs = random.sample(range(n_images), int(n_images * VALID_RATIO))

images_train = []
images_valid = []
image_train_id = 0
image_valid_id = 0
annotations_train = []
annotations_valid = []
for image_idx, frame_ref in enumerate(dict_images):
    print("Parsing image " + str(image_idx) + " of " + str(len(dict_images)))
    image_data = dict_images[frame_ref]
    if image_idx in valid_idxs:
        image_valid_id += 1
        image_data["id"] = image_valid_id
        if EXPORT_YOLO:
            yolo_annotations_file = open(dataset_output_valid_images_path + "/" + image_data["file_name"][:-4] + ".txt", "w")
        for annotation in image_data["annotations"]:
            annotation["id"] = len(annotations_valid) + 1
            annotation["image_id"] = image_valid_id
            annotations_valid.append(annotation)
            if EXPORT_YOLO and (yolo_annotations_file is not None):
                if annotation["category_id"] > len(CLASS_NAMES):
                    pass
                yolo_annotations_file.write("%d %f.6 %f.6 %f.6 %f.6\n" % (
                    annotation["category_id"] - 1,
                    (annotation["bbox"][0] + annotation["bbox"][2] / 2) / image_data["width"],
                    (annotation["bbox"][1] + annotation["bbox"][3] / 2) / image_data["height"],
                    annotation["bbox"][2] / image_data["width"],
                    annotation["bbox"][3] / image_data["height"]
                ))
        if EXPORT_YOLO and (yolo_annotations_file is not None):
            yolo_annotations_file.close()
        image_data.pop("annotations")
        #frame_path = images_source_path + "/" + image_data["file_name"]
        frame_path = findParsedFile(image_data["file_name"])
        try:
            shutil.copy(frame_path, dataset_output_valid_images_path)
        except:
            print("Failed to copy frame " + frame_path)
        images_valid.append(image_data)
    else:
        image_train_id += 1
        image_data["id"] = image_train_id
        if EXPORT_YOLO:
            yolo_annotations_file = open(dataset_output_train_images_path + "/" + image_data["file_name"][:-4] + ".txt", "w")
        for annotation in image_data["annotations"]:
            annotation["id"] = len(annotations_train) + 1
            annotation["image_id"] = image_train_id
            annotations_train.append(annotation)
            if EXPORT_YOLO and (yolo_annotations_file is not None):
                if annotation["category_id"] > len(CLASS_NAMES):
                    pass
                yolo_annotations_file.write("%d %f.6 %f.6 %f.6 %f.6\n" % (
                    annotation["category_id"] - 1,
                    (annotation["bbox"][0] + annotation["bbox"][2] / 2) / image_data["width"],
                    (annotation["bbox"][1] + annotation["bbox"][3] / 2) / image_data["height"],
                    annotation["bbox"][2] / image_data["width"],
                    annotation["bbox"][3] / image_data["height"]
                ))
        if EXPORT_YOLO and (yolo_annotations_file is not None):
            yolo_annotations_file.close()
        image_data.pop("annotations")
        #frame_path = images_source_path + "/" + image_data["file_name"]
        frame_path = findParsedFile(image_data["file_name"])
        try:
            shutil.copy(frame_path, dataset_output_train_images_path)
        except:
            print("Failed to copy frame " + frame_path)
        images_train.append(image_data)

'''for _, frame_ref in enumerate(dict_images):
    image_data = dict_images[frame_ref]
    images.append(image_data)'''

categories = []
category_name_list = []
label_list = []
for _, category_name in enumerate(dict_categories):
    category_id = dict_categories[category_name]
    # category_id = CATEGORIES_MAP[category_id - 1]
    # category_name = list(dict_categories.keys())[list(dict_categories.values()).index(category_id)]
    category = {"supercategory": "class", "id": category_id, "name": category_name}
    categories.append(category)
    category_name_list.append(category_name)
    label_list.append(str(category_id) + ": " + str(category_id))
category_names = '\'' + ('\', \''.join(category_name_list)) + '\''
label_map = ', '.join(label_list)

if EXPORT_YOLO:
    with open(dataset_output_path + "/" + "classes.txt", "w") as classes_file:
        for _, category_name in enumerate(dict_categories):
            classes_file.write("%s\n" % category_name)

licenses = [{"id": 0, "name": "Unknown License", "url": ""}]

info = {
    "year": 2020,
    "version": "1.0",
    "description": "Test",
    "contributor": "",
    "url": "Test",
    "date_created": "Mon Nov 02 2020 00:42:17 GMT+0300"
}

coco_train = {"info": info, "images": images_train, "annotations": annotations_train, "licenses": licenses, "categories": categories}
coco_valid = {"info": info, "images": images_valid, "annotations": annotations_valid, "licenses": licenses, "categories": categories}

coco_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

file_train = dataset_output_train_path + "/" + coco_timestamp + "_coco_train.json"
with open(file_train, 'w') as outfile:
    json.dump(coco_train, outfile, indent=4)

file_valid = dataset_output_valid_path + "/" + coco_timestamp + "_coco_valid.json"
with open(file_valid, 'w') as outfile:
    json.dump(coco_valid, outfile, indent=4)

with open(dataset_output_path + "/" + coco_timestamp + "_coco_config_add.py", "w") as config_add_py:
    config_add_py.write("%s_dataset = dataset_base.copy({\n" % OUTPUT_DATASET_NAME)
    config_add_py.write("    'name': '%s',\n" % OUTPUT_DATASET_NAME)
    config_add_py.write("    'train_info': '%s',\n" % file_train)
    config_add_py.write("    'train_images': '%s',\n" % dataset_output_train_images_path)
    config_add_py.write("    'valid_info': '%s',\n" % file_valid)
    config_add_py.write("    'valid_images': '%s',\n" % dataset_output_valid_images_path)
    config_add_py.write("    'class_names': (%s),\n" % category_names)
    config_add_py.write("    'label_map': { %s }\n" % label_map)
    config_add_py.write("})\n")
    config_add_py.write("\n")
    config_add_py.write("yolact_%s_config = yolact_resnet50_config.copy({\n" % OUTPUT_DATASET_NAME)
    config_add_py.write("    'name': 'yolact_%s',\n" % OUTPUT_DATASET_NAME)
    config_add_py.write("    'dataset': %s_dataset,\n" % OUTPUT_DATASET_NAME)
    #config_add_py.write("    'num_classes': %s,\n" % str(len(label_list)))
    config_add_py.write("    'num_classes': len(%s_dataset.class_names) + 1,\n" % OUTPUT_DATASET_NAME)
    config_add_py.write("    'max_size': %s\n" % str(MAX_IMAGE_SIZE))
    config_add_py.write("})")

'''heads_dataset = dataset_base.copy({
  'name': DATASET_NAME,
  'train_info': './data/heads/train/via_project_1Nov2020_16h10m_coco.json',
  'train_images': './data/heads/train/images/',
  'valid_info': './data/heads/val/via_project_1Nov2020_16h10m_coco.json',
  'valid_images': './data/heads/val/images/',
  'class_names': ('head'),
  'label_map': { 1:  1 }
})



yolact_heads_config = yolact_resnet50_config.copy({
    'name': 'yolact_' + DATASET_NAME,
    # Dataset stuff
    'dataset': heads_dataset,
    'num_classes': len(heads_dataset.class_names) + 1,

    # Image Size
    'max_size': 512,
})'''