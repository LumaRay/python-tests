import cv2
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
from PIL import ImageDraw

vcap = cv2.VideoCapture("rtsp://admin:LABCC0805$@192.168.1.64")#/Streaming/Channels/102")
#vcap = cv2.VideoCapture(0)#/Streaming/Channels/102")
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805$@192.168.1.64 latency=0 ! rtph265depay ! h265parse ! omxh265dec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
#vcap.set(cv2.CAP_PROP_FPS, 1)
vcap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
vcap.set(cv2.CAP_PROP_POS_FRAMES, 1)

engine = DetectionEngine("../face_detection/edge_tpu/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
#engine = DetectionEngine("ssd_mobilenet_v2_face_quant_postprocess.tflite")

while(1):

    ret, frame = vcap.read()

    if not ret:
        continue

    im_pil = Image.fromarray(frame)

    bboxes = engine.detect_with_image(im_pil,
                                    threshold=0.01,
                                    keep_aspect_ratio=True,
                                    relative_coord=False,
                                    top_k=10)

    for bbox in bboxes:
        x0 = int(bbox.bounding_box[0][0])
        y0 = int(bbox.bounding_box[0][1])
        x1 = int(bbox.bounding_box[1][0])
        y1 = int(bbox.bounding_box[1][1])
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 3)
    cv2.imshow('VIDEO', frame)
    cv2.waitKey(1)