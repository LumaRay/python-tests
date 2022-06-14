import json

GLOBAL_FLIP = False
# GLOBAL_FLIP = True

EFFECT_GRAY = "gray"
EFFECT_HSV = "hsv"
EFFECT_LAB = "lab"
EFFECT_YCC = "ycc"
EFFECT_COMPRESS_CUBIC = "comp"  # don't use
EFFECT_DECOMPRESS_CUBIC = "decomp"  # don't use
EFFECT_BLUR = "blur"
EFFECT_SHARPEN = "sharp"
EFFECT_BRIGHTEN = "bright"
EFFECT_DARKEN = "dark"
EFFECT_CONTRAST_INC = "cont"
EFFECT_CONTRAST_DEC = "decont"
EFFECT_SATURATE = "sat"
EFFECT_DESATURATE = "desat"
EFFECT_ADAPTIVE = "ada"
EFFECT_NORMALIZE = "norm"
EFFECT_DEVIATION = "dev"
EFFECT_DEVIATION2 = "devv"
EFFECT_TEST1 = "test1"
EFFECT_TEST2 = "test2"
EFFECT_FLIP = "flip"

PREP_SEQUENCES_LIST = [
    # [{"effect": EFFECT_GRAY, "value": 3}],
    # [{"effect": EFFECT_GRAY, "value": 0}],
    # [{"effect": EFFECT_HSV, "value": None}],
    # [{"effect": EFFECT_YCC, "value": None}],
    # [{"effect": EFFECT_LAB, "value": 3}],
    # [{"effect": EFFECT_LAB, "value": 0}],
    [{"effect": EFFECT_BLUR, "value": 9}],
    [{"effect": EFFECT_BLUR, "value": 3}],
    [{"effect": EFFECT_BLUR, "value": 1}],
    [{"effect": EFFECT_SHARPEN, "value": 9}],
    [{"effect": EFFECT_BRIGHTEN, "value": 10}],
    [{"effect": EFFECT_DARKEN, "value": 10}],
    [{"effect": EFFECT_CONTRAST_INC, "value": 10}],
    [{"effect": EFFECT_CONTRAST_DEC, "value": 10}],
    [{"effect": EFFECT_CONTRAST_DEC, "value": 6}],
    [{"effect": EFFECT_SATURATE, "value": 10}],
    [{"effect": EFFECT_DESATURATE, "value": 10}],
    # [{"effect": EFFECT_ADAPTIVE, "value": 3}],
    [{"effect": EFFECT_NORMALIZE, "value": ""}],
    [{"effect": EFFECT_DEVIATION, "value": ""}],
    [{"effect": EFFECT_DEVIATION2, "value": ""}],
    [{"effect": EFFECT_TEST1, "value": ""}],
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

EXTEND_PROGRAM = [
    {
        "images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images',
        "prep_sequences_list": PREP_SEQUENCES_LIST,
        "coco_path": "/../",
        "coco_json": "train_via_project_5Dec2020_13h12m_coco.json",
    },
    {
        "images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images',
        "prep_sequences_list": PREP_SEQUENCES_LIST,
        "coco_path": "/../",
        "coco_json": "val_via_project_23Dec2020_14h27m_coco.json",
    },
]

def formatEffectString(effect_type, effect_value):
    effect_print = effect_type
    if effect_value != "":
        effect_print += str(effect_value)
    return effect_print

for prep_set in EXTEND_PROGRAM:
    images_folder_src = prep_set["images_folder_src"]
    prep_sequences_list = prep_set["prep_sequences_list"]
    coco_json = prep_set["coco_json"]
    coco_path = prep_set["coco_path"]
    effect_folders = []
    for prep_sequence in prep_sequences_list:
        effect_folder = ""
        for prep in prep_sequence:
            effect_folder += formatEffectString(prep["effect"], prep["value"])
        # images_folder_dst = images_folder_src + '/' + effect_folder
        effect_folders.append(effect_folder)
    coco_json_path = images_folder_src + coco_path + coco_json
    # coco_output_path = images_folder_src + coco_output_path
    with open(coco_json_path) as json_file:
        coco_entry = json.load(json_file)
        new_images = []
        new_annotations = []
        for image in coco_entry["images"]:
            new_image = {**image}
            old_image_original_id = image["id"]
            new_image_original_id = len(new_images) + 1
            new_image["id"] = new_image_original_id
            new_images.append(new_image)
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
                    new_annotation = {**annotation}
                    new_annotation["image_id"] = new_image_original_id
                    new_annotation["id"] = len(new_annotations) + 1
                    new_annotations.append(new_annotation)
                    for new_image_id in new_images_ids:
                        new_annotation = {**annotation}
                        new_annotation["image_id"] = new_image_id
                        new_annotation["id"] = len(new_annotations) + 1
                        for new_flip_image in new_flip_images:
                            if new_image_id == new_flip_image["id"]:
                                new_annotation["bbox"][0] = new_flip_image["width"] - new_annotation["bbox"][0] - \
                                                            new_annotation["bbox"][2]
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
