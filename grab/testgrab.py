# import pyscreenshot as ImageGrab
# import cv2
# import numpy as np
#
# while True:
#     im = ImageGrab.grab()
#     im2 = np.asanyarray(im)
#
#     im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)
#     cv2.imshow("test", im2)
#
#     k = cv2.waitKey(10)
#     if k == 27:         # wait for ESC key to exit
#         cv2.destroyAllWindows()

# cv2.waitKey(0)

EXPORT_IMGS = False
# EXPORT_IMGS = True
EXPORT_STEP = 2
EXPORT_COUNT = 100

# USE_INFERENCE = False
USE_INFERENCE = True
MODEL_PATH = "/home/thermalview/Desktop/ThermalView/tests/grab/simple-line-10s-span10min-es2/2021-05-14-16-51-02-391741/gpaph_2021-05-14-16-51-02-3917410.207.h5"
# MODEL_PATH = "/home/thermalview/Desktop/ThermalView/tests/grab/simple-line-10s-span10min-es2/2021-05-14-17-52-48-141072/gpaph_2021-05-14-17-52-48-1410720.209.h5"

import time
from mss import mss
# from PIL import Image
import numpy as np
import cv2
from datetime import datetime
import os
import pathlib
pathToScriptFolder = str(pathlib.Path().absolute())

if USE_INFERENCE:
    from tensorflow import keras
    import tensorflow as tf
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model = keras.models.load_model(MODEL_PATH)

start_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

# EXPORT_FOLDER = pathToScriptFolder + "/frames_src/pro-10s-ma5-ma10-ma30/" + start_timestamp + "/"
EXPORT_FOLDER = pathToScriptFolder + f"/frames_src/simple-line-10s-span10min-es{EXPORT_STEP}/" + start_timestamp + "/"

if not os.path.exists(EXPORT_FOLDER):
    os.makedirs(EXPORT_FOLDER)

def capture_screenshot():
    # Capture entire screen
    with mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        im2 = np.asarray(sct_img)[:, :, :-1]
        return im2
        # Convert to PIL/Pillow Image
        # return Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')

if EXPORT_IMGS:
    time.sleep(5)

frame_idx = 0

last_pred = None

while True:
    img = capture_screenshot()
    frame_idx += 1
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_chart = img[300:300+300, 650:650+600]
    img_sell = img[390:390 + 320, 1430:1430 + 170]
    img_buy = img[760:760+310, 1430:1430+170]
    img_hist = img[300:300+710, 1730:1730+170]

    if EXPORT_IMGS:
        frame_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        cv2.imwrite(f"{EXPORT_FOLDER}{frame_timestamp}_chart.jpg", img_chart)
        cv2.imwrite(f"{EXPORT_FOLDER}{frame_timestamp}_sell.jpg", img_sell)
        cv2.imwrite(f"{EXPORT_FOLDER}{frame_timestamp}_buy.jpg", img_buy)
        cv2.imwrite(f"{EXPORT_FOLDER}{frame_timestamp}_hist.jpg", img_hist)
    else:
        cv2.imshow("chart", img_chart)
        cv2.imshow("sell", img_sell)
        cv2.imshow("buy", img_buy)
        cv2.imshow("hist", img_hist)
        if USE_INFERENCE:
            img_buy = cv2.normalize(img_buy, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img_sell = cv2.normalize(img_sell, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img_hist = cv2.normalize(img_hist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img_chart = cv2.normalize(img_chart, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img_buy = np.where(img_buy == 1, 1, 0)
            img_sell = np.where(img_sell == 1, 1, 0)
            img_hist = np.where(img_hist == 1, 1, 0)
            img_chart = np.where(img_chart == 1, 1, 0)
            # predictions = model.predict([np.moveaxis(img_buy, -1, 2), np.moveaxis(img_sell, -1, 2), np.moveaxis(img_hist, -1, 2), np.moveaxis(img_chart, -1, 2)])
            img_buy = np.moveaxis(img_buy, -1, 0)
            img_sell = np.moveaxis(img_sell, -1, 0)
            img_hist = np.moveaxis(img_hist, -1, 0)
            img_chart = np.moveaxis(img_chart, -1, 0)
            img_buy = np.expand_dims(img_buy, 0)
            img_sell = np.expand_dims(img_sell, 0)
            img_hist = np.expand_dims(img_hist, 0)
            img_chart = np.expand_dims(img_chart, 0)
            img_buy = np.expand_dims(img_buy, 0)
            img_sell = np.expand_dims(img_sell, 0)
            img_hist = np.expand_dims(img_hist, 0)
            img_chart = np.expand_dims(img_chart, 0)
            predictions = model.predict([img_buy, img_sell, img_hist, img_chart])
            predictions[0] = cv2.normalize(predictions[0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            out_img = (predictions[0] * 255).astype(np.uint8)
            if last_pred is not None:
                pred_diff = out_img - last_pred
                print(np.count_nonzero(pred_diff))
            last_pred = out_img
            cv2.imshow("test-out", out_img)

    k = cv2.waitKey(1)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
        break

    if EXPORT_IMGS:
        if frame_idx >= EXPORT_COUNT:
            break
        time.sleep(EXPORT_STEP)

# img.show()