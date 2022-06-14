mkdir train
mkdir valid
mkdir test
cp ~/Desktop/ThermalView/tests/yolo4_tiny_train/classes.txt train/_darknet.labels
cp ~/Desktop/ThermalView/tests/yolo4_tiny_train/test-yolo1.jpg train/
cp ~/Desktop/ThermalView/tests/yolo4_tiny_train/test-yolo1.txt train/
cp ~/Desktop/ThermalView/tests/yolo4_tiny_train/classes.txt valid/_darknet.labels
cp ~/Desktop/ThermalView/tests/yolo4_tiny_train/test-yolo1.jpg valid/
cp ~/Desktop/ThermalView/tests/yolo4_tiny_train/test-yolo1.txt valid/
cp ~/Desktop/ThermalView/tests/yolo4_tiny_train/classes.txt test/_darknet.labels
cp ~/Desktop/ThermalView/tests/yolo4_tiny_train/test-yolo1.jpg test/
cp ~/Desktop/ThermalView/tests/yolo4_tiny_train/test-yolo1.txt test/

cp train/_darknet.labels data/obj.names
mkdir data/obj
cp train/*.jpg data/obj/
cp valid/*.jpg data/obj/
cp train/*.txt data/obj/
cp valid/*.txt data/obj/

python3 ./train.py

#num_classes = 4
#max_batches = 8000
#steps1 = 6400
#steps2 = 7200
#steps_str = 6400,7200
#num_filters = 27

cp ~/Desktop/ThermalView/tests/yolo4_tiny_train/custom-yolov4-tiny-detector.cfg cfg/

./darknet detector train data/obj.data cfg/custom-yolov4-tiny-detector.cfg yolov4-tiny.conv.29 -dont_show -map

cp ~/darknet/backup/custom-yolov4-tiny-detector_last.weights ~/Desktop/ThermalView/face_detection/yolov4_tiny/

cd ~/Desktop/ThermalView/face_detection/yolov4_tiny/

python3 ./convert_weights_pb.py --class_names ./obj.names --weights_file custom-yolov4-tiny-detector_last.weights --tiny --output_graph custom-yolov4-tiny-detector_last.pb

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model custom-yolov4-tiny-detector_last.pb --transformations_config custom_yolo_v4_tiny.json --batch 1 --data_type FP16

git rm --cached custom-yolov4-tiny-detector_last.bin
git add custom-yolov4-tiny-detector_last.bin