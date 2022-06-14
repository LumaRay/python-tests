import time

from edgetpu.detection.engine import DetectionEngine

face_detection_engine = DetectionEngine(FACE_DETECTION_MODEL_PATH)

from imutils.video import VideoStream

video_stream = VideoStream(src=0).start()
time.sleep(1.0)

import cv2

while True:
    input_frame = cv2.flip(video_stream.read(), 1)
    frame_as_image, resized_frame = frame_processor.preprocess(input_frame)

import imutils
from PIL import Image

def preprocess(frame):
    resized_frame = imutils.resize(frame, width=IMAGE_WIDTH)
    rgb_array = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    frame_as_image = Image.fromarray(rgb_array)
    return frame_as_image, resized_frame

face_detection_engine.detect_with_image(
    frame_as_image,
    threshold=confidence,
    keep_aspect_ratio=True,
    relative_coord=False,
    top_k=MAX_FACES,
)

for face in detected_faces:
    bounding_box = face.bounding_box.flatten().astype("int")
    face_filter = cache.update(bounding_box)
    frame = frame_processor.replace_face(
        bounding_box, resized_frame , face_filter
    )

    (bbox_x1, bbox_y1, bbox_x2, bbox_y2) = bounding_box
    width = bbox_x2 - bbox_x1
    height = bbox_y2 - bbox_y1
    face_filter_resized = cv2.resize(
        face_filter, (width, height), interpolation=cv2.INTER_AREA
    )

    face_filter_alpha = face_filter[:, :, 3] / 255.0
    inverted_alpha = 1.0 - face_filter_alpha
    for colour_index in range(0, 3):
        frame[bbox_y1:bbox_y2, bbox_x1:bbox_x2, colour_index] = (
                face_filter_alpha * face_filter[:, :, colour_index]
                + inverted_alpha * frame[bbox_y1:bbox_y2, bbox_x1:bbox_x2, colour_index]
        )

    cv2.imshow(window_name, frame)