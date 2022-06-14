import json
import os
import pathlib
import shutil

pathToScriptFolder = str(pathlib.Path().absolute().as_posix())

DATASET_NAMES = [
#     'maskfacesnew',
#     'maskfacesnew3',
    'maskfacesnewwork1',
    'maskfacesnewwork2',
#     'maskfacesnewwork_toadd1',
    'false_faces_work1',
    'source_2020_11_25_1000ms',
    'source_2020_11_25_1000ms5max',
    'source_2020_11_25_200ms',
    'source_2020_11_25_3000ms',
    'source_2020_11_25_3000ms5max',
    'source_2020_11_25_300ms',
    'source_2020_11_25_noms',
#     'vk_sources1',
    'v2_source_2020_11_17',
    'v2_source_2020_11_19',
    'v2_source_2020_11_25_1000ms',
    'v2_source_2020_11_25_1000ms5max',
    'v2_source_2020_11_25_200ms',
    'v2_source_2020_11_25_3000ms',
    'v2_source_2020_11_25_3000ms5max',
    'v2_source_2020_11_25_300ms',
    'v2_source_2020_11_25_noms',
]

CLASS_NAMES = ['mask', 'maskchin', 'masknone', 'masknose']
CLASSES_MAP = [0, 1, 2, 3]  # All classes
# CLASSES_MAP = [0, 2, 2, 2]  # mask/masknone classes

# CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']
# CLASSES_MAP = [0, 1, 2, 3, 4]  # All classes

# CLASS_NAMES = ['back', 'mask', 'maskchin', 'masknone', 'masknose']
# CLASSES_MAP = [0, 0, 0, 0, 0]  # All classes

# OUTPUT_DATASET_NAME = 'maskfacesnewwork0312added1ffw1a1vk1exp1nb'
# OUTPUT_DATASET_NAME = 'maskfacesnewwork12ffw1a1nbgray'
OUTPUT_DATASET_NAME = 'maskfacesnewwork12ffw1a1v12nbgray'
# OUTPUT_DATASET_NAME = 'headsnew'

dataset_sources_path = pathToScriptFolder + '/../dataset_sources'

parsed_data_path = pathToScriptFolder + '/../parsed_data'
parsed_data_last_path_list = []

dataset_output_path = pathToScriptFolder + '/../dataset_outputs/yolo/' + OUTPUT_DATASET_NAME
dataset_output_images_path = dataset_output_path + '/images'

if not os.path.exists(dataset_output_images_path):
    os.makedirs(dataset_output_images_path)

dict_images = {}

dict_categories = {}

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

for class_idx, class_name in enumerate(CLASS_NAMES):
    print("Parsing class " + class_name)
    for dataset_name in DATASET_NAMES:
        print("Parsing dataset " + dataset_name)
        class_path = dataset_sources_path + '/' + dataset_name + '/' + class_name
        new_class_idx = CLASSES_MAP[class_idx]
        new_class_name = CLASS_NAMES[new_class_idx]
        print("Mapping class {} to {}".format(class_name, new_class_name))
        class_name = new_class_name
        if dict_categories.get(class_name) is None:
            category_id = len(dict_categories)
            dict_categories[class_name] = category_id
        else:
            category_id = dict_categories[class_name]
        files = os.listdir(class_path)
        for fidx, file in enumerate(files):
            if file.endswith(".jpg"):
                file_name, file_ext = file.split('.')
                frame_timestamp, score, timestamp = file_name.split('_')
                object_path = findParsedFile(file_name + '.txt')
                print("Parsing class=", class_name, "dataset=", dataset_name, "file", fidx, "of", len(files), file_name, "object_path=", object_path)
                if object_path is None:
                    input("Press Enter to continue...")
                    continue
                with open(object_path) as json_file:
                    entry = json.load(json_file)
                    frame_ref = entry['frame_ref']
                    frame_width = entry['frame_width']
                    frame_height = entry['frame_height']
                    bbox = entry['bbox']
                    bbox_x_rel = bbox['x'] / frame_width
                    bbox_y_rel = bbox['y'] / frame_height
                    bbox_w_rel = bbox['w'] / frame_width
                    bbox_h_rel = bbox['h'] / frame_height
                    bbox_x_center_rel = bbox_x_rel + bbox_w_rel / 2
                    bbox_y_center_rel = bbox_y_rel + bbox_h_rel / 2
                    if dict_images.get(frame_ref) is None:
                        dict_images[frame_ref] = {"frame_ref": frame_ref, "annotations": []}
                    dict_images[frame_ref]["annotations"].append({
                        "bbox_rel": [
                            bbox_x_center_rel,
                            bbox_y_center_rel,
                            bbox_w_rel,
                            bbox_h_rel
                        ],
                        "category_id": category_id,
                    })

for image_id, (frame_ref, image_data) in enumerate(dict_images.items(), 1):
    print("Parsing image " + str(image_id) + " of " + str(len(dict_images)))
    with open(dataset_output_images_path + "/" + image_data["frame_ref"] + ".txt", "w") as annotations_file:
        for annotation in image_data["annotations"]:
            annotations_file.write("%d %.6f %.6f %.6f %.6f\n" % (
                annotation["category_id"],
                annotation["bbox_rel"][0],
                annotation["bbox_rel"][1],
                annotation["bbox_rel"][2],
                annotation["bbox_rel"][3]
            ))
    frame_path = findParsedFile(image_data["frame_ref"] + ".jpg")
    try:
        shutil.copy(frame_path, dataset_output_images_path)
    except:
        print("Failed to copy frame " + frame_path)

with open(dataset_output_path + "/" + "classes.txt", "w") as classes_file:
    for _, category_name in enumerate(dict_categories):
        classes_file.write("%s\n" % category_name)

