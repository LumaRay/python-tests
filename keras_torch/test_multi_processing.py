# import multiprocessing as mp
# import torch.multiprocessing as mp
from torch.multiprocessing import Process
# from torch.multiprocessing.spawn import freeze_support
import time

def processUemMask():
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_GPU_MEMORY_LIMIT_MB = 128 # 256 # 512 # 1024  # 2048
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(gpu, [
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=TENSORFLOW_GPU_MEMORY_LIMIT_MB)])
        except RuntimeError as e:
            print(e)
    uem_mask_model = load_model(
        "/home/thermalview/Desktop/ThermalView/mask_detection/uem_mask/ta946tl180_i_va0.9989517vl0.0014108_r01e0001_allw12vk2p1randd5v20-1-20-0-128l256l512l728l1024-128x128x3-sn.h5")
    in_frames_array = np.zeros((10, 128, 128, 3), dtype=np.uint8)
    start_time = round(time.monotonic() * 1000)
    while True:
        time.sleep(0.0010)
        time_taken_idle = round(time.monotonic() * 1000) - start_time
        # print("uem_mask_model.predict loop idle", str(time_taken_idle), "ms")  #
        start_time = round(time.monotonic() * 1000)
        predictions = uem_mask_model.predict(in_frames_array)
        time_taken = round(time.monotonic() * 1000) - start_time
        print("uem_mask_model.predict loop", time_taken, "+", time_taken_idle, "=", time_taken + time_taken_idle, "ms")  # 25-37
        start_time = round(time.monotonic() * 1000)

def processFacenet512():
    import numpy as np
    import tensorflow as tf
    TENSORFLOW_GPU_MEMORY_LIMIT_MB = 512 # 1024  # 2048
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(gpu, [
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=TENSORFLOW_GPU_MEMORY_LIMIT_MB)])
        except RuntimeError as e:
            print(e)
    from face_recognition_modules.deepface.DeepFace import build_model
    facenet512_model = build_model("Facenet512")
    # facenet512_model = Facenet512.loadModel("/home/thermalview/Desktop/ThermalView/face_recognition_modules/deepface/weights/facenet512_weights.h5")
    # facenet512_model = facenet512_model()
    # Facenet512.loadModel("/home/thermalview/Desktop/ThermalView/face_recognition_modules/deepface/weights/facenet512_weights.h5")
    in_frames_array = np.zeros((10, 160, 160, 3), dtype=np.uint8)
    start_time = round(time.monotonic() * 1000)
    while True:
        time.sleep(0.0010)
        time_taken_idle = round(time.monotonic() * 1000) - start_time
        start_time = round(time.monotonic() * 1000)
        predictions = facenet512_model.predict(in_frames_array)
        time_taken = round(time.monotonic() * 1000) - start_time
        print("facenet512_model.predict loop", time_taken, "+", time_taken_idle, "=", time_taken + time_taken_idle, "ms")  # 33-40
        start_time = round(time.monotonic() * 1000)

def processYolact():
    import numpy as np
    from face_detection.yolact.yolact import Yolact
    # THCudaCheck FAIL file=/pytorch/aten/src/THC/THCGeneral.cpp line=47 error=804 : forward compatibility was attempted on non supported HW
    # https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch/45319156#45319156
    #
    import torch
    import torch.backends.cudnn as cudnn
    from face_detection.yolact.utils.functions import SavePath as yolact_SavePath
    from face_detection.yolact.config import cfg as yolact_cfg, set_cfg as yolact_set_cfg, COLORS as yolact_COLORS
    from face_detection.yolact.utils.augmentations import FastBaseTransform as yolact_FastBaseTransform
    # from face_detection.yolact.utils.functions import MovingAverage
    # from face_detection.yolact.layers.output_utils import postprocess as yolact_postprocess, undo_image_transformation as yolact_undo_image_transformation
    # from collections import defaultdict

    # yolact_color_cache = defaultdict(lambda: {})
    yolact_trained_model = '/home/thermalview/Desktop/ThermalView/face_detection/yolact/weights/yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbgray640_64_146000.pth'  # n845/349 849/354 852/353 499:851/353 854/347 (830/388 833/383 831/393 828/391)

    yolact_cuda = True
    # yolact_top_k = 150
    yolact_fast_nms = True
    yolact_cross_class_nms = False
    # yolact_display_masks = True
    # yolact_display_bboxes = True
    # yolact_display_text = True
    # yolact_display_scores = True
    # yolact_display_lincomb = False
    yolact_mask_proto_debug = False
    yolact_video_multiframe = 1
    # yolact_score_threshold = 0.015
    # yolact_crop = True

    yolact_model_path = yolact_SavePath.from_str(yolact_trained_model)
    yolact_config = yolact_model_path.model_name + '_config'
    yolact_set_cfg(yolact_config)

    with torch.no_grad():
        if yolact_cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        yolact_net = Yolact()
        yolact_net.load_weights(yolact_trained_model)
        yolact_net.eval()

        if yolact_cuda:
            yolact_net = yolact_net.cuda()

        yolact_net.detect.use_fast_nms = yolact_fast_nms
        yolact_net.detect.use_cross_class_nms = yolact_cross_class_nms
        yolact_cfg.mask_proto_debug = yolact_mask_proto_debug

        class yolact_CustomDataParallel(torch.nn.DataParallel):
            """ A Custom Data Parallel class that properly gathers lists of dictionaries. """

            def gather(self, outputs, output_device):
                # Note that I don't actually want to convert everything to the output_device
                return sum(outputs, [])

        yolact_net = yolact_CustomDataParallel(yolact_net).cuda()
        yolact_transform = torch.nn.DataParallel(yolact_FastBaseTransform()).cuda()

        img = np.zeros((720, 1280, 3))
        frames = [img]
        if yolact_cuda:
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
        else:
            frames = [torch.from_numpy(frame).float() for frame in frames]
        imgs = yolact_transform(torch.stack(frames, 0))
        num_extra = 0
        while imgs.size(0) < yolact_video_multiframe:
            imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
            num_extra += 1

        start_time = round(time.monotonic() * 1000)
        while True:
            time.sleep(0.0010)
            time_taken_idle = round(time.monotonic() * 1000) - start_time
            start_time = round(time.monotonic() * 1000)
            out = yolact_net(imgs)
            time_taken = round(time.monotonic() * 1000) - start_time
            print("yolact_net(imgs) loop", time_taken, "+", time_taken_idle, "=", time_taken + time_taken_idle, "ms")  # 77-84
            start_time = round(time.monotonic() * 1000)

if __name__ == '__main__':
    # freeze_support()
    procUemMask = Process(target=processUemMask)
    procFacenet512 = Process(target=processFacenet512)
    procYolact = Process(target=processYolact)

    procUemMask.start()
    procFacenet512.start()
    procYolact.start()

    procUemMask.join()
    procFacenet512.join()
    procYolact.join()


# uem_mask+facenet512+yolact: RAM13.8 GPURAM4.9(0.1GBerr+0.5GBerr) GPU93 CPU100

# uem_mask+facenet512: RAM10.3 GPURAM5.6(2GB)->3.6(1GB)->2.6(0.5GBerr)->2.3(0.1GBerr+0.5GBerr) GPU38 CPU100

'''uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 33 + 1 = 34 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
uem_mask_model.predict loop 29 + 1 = 30 ms
facenet512_model.predict loop 36 + 1 = 37 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 35 + 2 = 37 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 36 + 1 = 37 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
facenet512_model.predict loop 34 + 1 = 35 ms
uem_mask_model.predict loop 30 + 2 = 32 ms
uem_mask_model.predict loop 28 + 1 = 29 ms
facenet512_model.predict loop 35 + 2 = 37 ms
uem_mask_model.predict loop 25 + 2 = 27 ms
facenet512_model.predict loop 33 + 1 = 34 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
uem_mask_model.predict loop 30 + 1 = 31 ms
facenet512_model.predict loop 38 + 1 = 39 ms
uem_mask_model.predict loop 25 + 2 = 27 ms
facenet512_model.predict loop 40 + 1 = 41 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 27 + 1 = 28 ms
uem_mask_model.predict loop 27 + 1 = 28 ms
facenet512_model.predict loop 36 + 1 = 37 ms
uem_mask_model.predict loop 29 + 1 = 30 ms
facenet512_model.predict loop 33 + 1 = 34 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
uem_mask_model.predict loop 28 + 1 = 29 ms
facenet512_model.predict loop 38 + 1 = 39 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
facenet512_model.predict loop 36 + 1 = 37 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 34 + 1 = 35 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
uem_mask_model.predict loop 27 + 1 = 28 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
facenet512_model.predict loop 37 + 1 = 38 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 39 + 1 = 40 ms
uem_mask_model.predict loop 32 + 1 = 33 ms
uem_mask_model.predict loop 28 + 1 = 29 ms
facenet512_model.predict loop 39 + 1 = 40 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 27 + 1 = 28 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 36 + 1 = 37 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
facenet512_model.predict loop 36 + 1 = 37 ms
uem_mask_model.predict loop 29 + 1 = 30 ms
uem_mask_model.predict loop 29 + 1 = 30 ms
facenet512_model.predict loop 38 + 1 = 39 ms
uem_mask_model.predict loop 29 + 1 = 30 ms
facenet512_model.predict loop 34 + 1 = 35 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 34 + 1 = 35 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
uem_mask_model.predict loop 28 + 1 = 29 ms
facenet512_model.predict loop 40 + 1 = 41 ms
uem_mask_model.predict loop 28 + 1 = 29 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 36 + 1 = 37 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
uem_mask_model.predict loop 29 + 1 = 30 ms
facenet512_model.predict loop 37 + 1 = 38 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
facenet512_model.predict loop 33 + 1 = 34 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 34 + 1 = 35 ms
uem_mask_model.predict loop 29 + 1 = 30 ms
facenet512_model.predict loop 36 + 1 = 37 ms
uem_mask_model.predict loop 29 + 2 = 31 ms
uem_mask_model.predict loop 31 + 1 = 32 ms
facenet512_model.predict loop 38 + 1 = 39 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 34 + 1 = 35 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
facenet512_model.predict loop 38 + 1 = 39 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
facenet512_model.predict loop 34 + 1 = 35 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
uem_mask_model.predict loop 28 + 1 = 29 ms
facenet512_model.predict loop 38 + 1 = 39 ms
uem_mask_model.predict loop 34 + 1 = 35 ms
facenet512_model.predict loop 37 + 1 = 38 ms
facenet512_model.predict loop 37 + 1 = 38 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 88 + 1 = 89 ms
facenet512_model.predict loop 33 + 1 = 34 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
uem_mask_model.predict loop 28 + 1 = 29 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 34 + 2 = 36 ms
uem_mask_model.predict loop 27 + 1 = 28 ms
facenet512_model.predict loop 42 + 1 = 43 ms
uem_mask_model.predict loop 34 + 1 = 35 ms
uem_mask_model.predict loop 30 + 1 = 31 ms
facenet512_model.predict loop 40 + 1 = 41 ms
uem_mask_model.predict loop 27 + 1 = 28 ms
facenet512_model.predict loop 34 + 2 = 36 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 28 + 1 = 29 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 29 + 1 = 30 ms
uem_mask_model.predict loop 28 + 1 = 29 ms
facenet512_model.predict loop 37 + 1 = 38 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 36 + 1 = 37 ms
uem_mask_model.predict loop 26 + 2 = 28 ms
facenet512_model.predict loop 34 + 1 = 35 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
uem_mask_model.predict loop 29 + 1 = 30 ms
facenet512_model.predict loop 34 + 1 = 35 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
facenet512_model.predict loop 33 + 1 = 34 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
facenet512_model.predict loop 34 + 1 = 35 ms
uem_mask_model.predict loop 27 + 1 = 28 ms
facenet512_model.predict loop 39 + 2 = 41 ms
uem_mask_model.predict loop 33 + 1 = 34 ms
uem_mask_model.predict loop 29 + 1 = 30 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 25 + 2 = 27 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
uem_mask_model.predict loop 31 + 1 = 32 ms
facenet512_model.predict loop 34 + 1 = 35 ms
uem_mask_model.predict loop 33 + 1 = 34 ms
facenet512_model.predict loop 37 + 1 = 38 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 28 + 1 = 29 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 33 + 1 = 34 ms
uem_mask_model.predict loop 29 + 1 = 30 ms
uem_mask_model.predict loop 28 + 1 = 29 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
facenet512_model.predict loop 37 + 1 = 38 ms
uem_mask_model.predict loop 25 + 1 = 26 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 27 + 1 = 28 ms
uem_mask_model.predict loop 28 + 1 = 29 ms
facenet512_model.predict loop 39 + 1 = 40 ms
uem_mask_model.predict loop 25 + 2 = 27 ms
facenet512_model.predict loop 37 + 1 = 38 ms
uem_mask_model.predict loop 29 + 1 = 30 ms
facenet512_model.predict loop 35 + 1 = 36 ms
uem_mask_model.predict loop 26 + 1 = 27 ms
uem_mask_model.predict loop 30 + 1 = 31 ms
facenet512_model.predict loop 36 + 1 = 37 ms
uem_mask_model.predict loop 30 + 1 = 31 ms
facenet512_model.predict loop 35 + 1 = 36 ms'''