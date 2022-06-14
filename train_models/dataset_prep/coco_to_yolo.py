import json

CONVERTION_PROGRAM = [
    {
        "images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/train/images',
        "coco_input": "/../train_via_project_5Dec2020_13h12m_coco.json",
        "yolo_classes": "/../../classes.txt",
    },
    {
        "images_folder_src": 'c:/Work/InfraredCamera/ThermalView/tests/train_models/yolact_train/facemasknoses/val/images',
        "coco_input": "/../val_via_project_23Dec2020_14h27m_coco.json",
        "yolo_classes": "/../../classes.txt",
    },
]

for prep_set in CONVERTION_PROGRAM:
    images_folder_src = prep_set["images_folder_src"]
    coco_input = prep_set["coco_input"]
    coco_input_path = images_folder_src + coco_input
    yolo_classes = prep_set["yolo_classes"]
    yolo_classes_path = images_folder_src + yolo_classes
    with open(coco_input_path) as json_file:
        coco_entry = json.load(json_file)
        for image in coco_entry["images"]:
            image_id = image["id"]
            image_width = image["width"]
            image_height = image["height"]
            file_name, file_ext = image["file_name"].split('.')
            with open(images_folder_src + "/" + file_name + ".txt", "w") as annotations_file:
                for annotation in coco_entry["annotations"]:
                    if annotation["image_id"] == image_id:
                        annotations_file.write("%d %.6f %.6f %.6f %.6f\n" % (
                            annotation["category_id"] - 1,
                            (annotation["bbox"][0] + annotation["bbox"][2] / 2) / image_width,
                            (annotation["bbox"][1] + annotation["bbox"][3] / 2) / image_height,
                            annotation["bbox"][2] / image_width,
                            annotation["bbox"][3] / image_height
                        ))
        with open(yolo_classes_path, "w") as classes_file:
            for category in coco_entry["categories"]:
                classes_file.write("%s\n" % category["name"])
