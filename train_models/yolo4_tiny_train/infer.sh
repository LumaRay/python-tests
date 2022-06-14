#!/bin/bash

python3 ./infer.py
ls backup
cp data/obj.names data/coco.names
./darknet detect cfg/custom-yolov4-tiny-detector.cfg backup/custom-yolov4-tiny-detector_best.weights "test/0" -dont-show
