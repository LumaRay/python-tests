# YOUTUBE_URL = 'https://www.youtube.com/watch?v=HC9FDfuUpKQ'
# YOUTUBE_URL = 'https://www.youtube.com/watch?v=VOCNWyy3zJU'
# YOUTUBE_URL = 'https://www.youtube.com/watch?v=a9TGMASZMJE'
YOUTUBE_URL = 'https://www.youtube.com/watch?v=KNM8w4kFiB0'

# USE_GPU_GRAB = True
USE_GPU_GRAB = False

# FRAME_SIZE = (720, 1280)
FRAME_SIZE = (1080, 1920)
# FRAME_SIZE = (1088, 1920)

FRAME_CHANNELS = 4
FACE_BBOXES_SHAPE = (256, 2, 4)
FACE_SIZE_DETECT_MASK = (128, 128)
FACE_SIZE_FIND_FEATURES = (160, 160)

# FACE_FEATURES_BASE_SIZE = 10_000
FACE_FEATURES_BASE_SIZE = 30_000
# FACE_FEATURES_BASE_SIZE = 100_000
# FACE_FEATURES_BASE_SIZE = 400_000
FACE_FEATURES_COUNT = 512

USE_MEMMAP = True
# USE_MEMMAP = False

USE_SYNC = False
# USE_SYNC = True

import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
# sys.path.append('/home/thermalview2/ThermalView')
sys.path.append('/home/thermalview/Desktop/ThermalView')

class FaceData:
    ignore = False
    show_bbox = True
    bbox_relativeOriginal = 0, 0, 0, 0
    bbox_relativeNormalOriginal = 0, 0, 0, 0
    bbox_relativeInfraredOriginal = 0, 0, 0, 0
    bbox_relativeInfraredOriginalScaled = 0, 0, 0, 0
    bbox_relativeDepthOriginal = 0, 0, 0, 0
    bbox_relative = 0, 0, 0, 0
    bbox_relativeNormal = 0, 0, 0, 0
    bbox_relativeInfrared = 0, 0, 0, 0
    bbox_relativeDepth = 0, 0, 0, 0
    face_mask = 1  # 0
    shape_roi_bgr_wide_resized = None
    shape_mask = None
    shape_roi_mask = None
    shape_roi_mask_wide = None
    shape_roi_mask_wide_resized = None
    distance = 0
    face_elements = {}
    temp_correction = 0
    max_temperature = 0
    smoothed_temperature = 0
    distance_index = -1
    clothes_temp = 0
    clothes_temp_index = -1
    heat_mass_index = -1
    temp_difference = 0
    track_id = None
    face_image = None
    face_image_wide = None
    face_image_shaped = None
    face_image_shaped_wide = None
    face_recognition_source = None
    face_features = None
    face_features_distance_min = None
    face_features_similar_group_idx = None
    face_features_similar_group_distance = None
    face_features_similar_group_verified_idx = None
    face_features_similar_group_verified_distance = None

def processGpuPipeline(dh_frame_placeholder, mp_face_bboxes_count, mp_face_bboxes, dh_face_features_base):
    import time

    TORCH_YOLACT_SHAPE = (640, 640)

    import numpy as np
    import cupy as cp
    import os

    os.system('taskset -p 0xffffffff %d' % os.getpid())

    from face_detection.yolact.yolact import Yolact
    import torch
    from torch.utils.dlpack import to_dlpack
    import torch.backends.cudnn as cudnn
    from face_detection.yolact.utils.functions import SavePath as yolact_SavePath
    from face_detection.yolact.config import cfg as yolact_cfg, set_cfg as yolact_set_cfg, COLORS as yolact_COLORS
    from face_detection.yolact.utils.augmentations import FastBaseTransform as yolact_FastBaseTransform
    from face_detection.yolact.layers.output_utils import postprocess as yolact_postprocess

    yolact_trained_model = '../../face_detection/yolact/weights/yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbgray640_64_146000.pth'  # n845/349 849/354 852/353 499:851/353 854/347 (830/388 833/383 831/393 828/391)

    yolact_fast_nms = True
    yolact_cross_class_nms = False
    yolact_display_lincomb = False
    yolact_mask_proto_debug = False
    yolact_crop = True

    score_threshold = 7 / 100  # self.settings.propFaceDetectionTorchYolactThreshold
    top_k = 150  # self.settings.propFaceDetectionTorchYolactTopK
    # YOLACT_SURFACE_AREA_MIN_THRESHOLD = 0.0006
    YOLACT_SURFACE_AREA_MIN_THRESHOLD = 0.0001

    yolact_model_path = yolact_SavePath.from_str(yolact_trained_model)
    yolact_config = yolact_model_path.model_name + '_config'
    yolact_set_cfg(yolact_config)

    with torch.no_grad():
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        yolact_net = Yolact()
        yolact_net.load_weights(yolact_trained_model)
        yolact_net.eval()

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
        yolact_extract_frame = lambda x, i: (
            x[0][i] if x[1][i]['detection'] is None else x[0][i].to(x[1][i]['detection']['box'].device), [x[1][i]])

    import imp
    with torch.no_grad():
        uem_mask_torch_weights = "../keras_torch/keras_to_torch_uem_mask_2.pt"  # path_to_the_numpy_weights
        A = imp.load_source('MainModel',
                            '../keras_torch/keras_to_torch_uem_mask_2.py')

        uem_mask_model = torch.load(uem_mask_torch_weights)
        # model = torch.load(torch_weights, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        uem_mask_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        uem_mask_model = uem_mask_model.to(uem_mask_device)
        uem_mask_model.eval()

    with torch.no_grad():
        # needs https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt
        # or https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt
        from face_recognition_modules.facenet_pytorch import InceptionResnetV1 as TimeslerInceptionResnetV1
        timesler_model = TimeslerInceptionResnetV1(pretrained='vggface2')
        timesler_model = timesler_model.eval()
        timesler_model.cuda()

    ca_frame_placeholder = dh_frame_placeholder.open()
    cp_frame_placeholder = cp.asarray(ca_frame_placeholder)

    np_face_bboxes = np.frombuffer(mp_face_bboxes.get_obj(), dtype=np.float32)
    np_face_bboxes = np_face_bboxes.reshape(FACE_BBOXES_SHAPE)

    ca_face_features_base = dh_face_features_base.open()
    cp_face_features_base = cp.asarray(ca_face_features_base)
    cp_diff_holder = cp.zeros_like(cp_face_features_base)
    cp_dist_holder = cp.zeros((512, FACE_FEATURES_BASE_SIZE), dtype=cp.float32)

    MEANS = (103.94, 116.78, 123.68)
    MEANS_REV = (123.68, 116.78, 103.94)
    STD = (57.38, 57.12, 58.40)
    STD_REV = (58.40, 57.12, 57.38)
    mean = torch.Tensor(MEANS).float().cuda()[None, :, None, None]
    mean_rev = torch.Tensor(MEANS_REV).float().cuda()[None, :, None, None]
    std = torch.Tensor(STD).float().cuda()[None, :, None, None]
    std_rev = torch.Tensor(STD_REV).float().cuda()[None, :, None, None]

    while True:
        time.sleep(0.00050)
        cp_frame_placeholder_copy = cp_frame_placeholder.copy()
        time.sleep(0.00050)
        start_time = round(time.monotonic() * 1000)
        cp_frame_gray = cp.dot(cp_frame_placeholder_copy[..., :3],
                               cp.asarray([0.2989, 0.5870, 0.1140], dtype=cp.float32)).astype(cp.uint8)
        # start_time = round(time.monotonic() * 1000)
        # cp_frame_gray = cp.repeat(cp_frame_gray[:, :, cp.newaxis], 3, axis=2)
        cp_frame_gray = cp.repeat(cp_frame_gray[:, :, cp.newaxis], 1, axis=2)
        # start_time = round(time.monotonic() * 1000)

        dlp_frame = cp_frame_gray.toDlpack()
        # start_time = round(time.monotonic() * 1000)
        dt_frame = torch.utils.dlpack.from_dlpack(dlp_frame)
        # start_time = round(time.monotonic() * 1000)

        dt_frame_yolact = dt_frame
        # np_frame = d_torch_yolact.cpu().numpy()
        dt_frame_yolact = dt_frame_yolact.unsqueeze(0)
        # start_time = round(time.monotonic() * 1000)
        # d_torch_src = d_torch_src.permute(0, 3, 1, 2)
        # d_torch_yolact = torch.nn.functional.interpolate(d_torch_yolact, TORCH_YOLACT_SHAPE)
        # np_frame = d_torch_yolact.cpu().numpy().squeeze(0)
        dt_frame_yolact = dt_frame_yolact.float()
        # start_time = round(time.monotonic() * 1000)
        # d_torch_yolact = d_torch_yolact / 255

        # np_frame = d_torch_yolact.cpu().numpy().squeeze(0)

        frame_faces_list = []
        frame_width = dt_frame_yolact.shape[1]
        frame_height = dt_frame_yolact.shape[0]
        num_extra = 0
        # d_torch_yolact = yolact_transform(torch.stack(d_torch_yolact, 0))

        # These are in BGR and are for ImageNet
        img = dt_frame_yolact.permute(0, 3, 1, 2)  # .contiguous()
        img = torch.nn.functional.interpolate(img, TORCH_YOLACT_SHAPE, mode='bilinear', align_corners=False)
        # img = img.repeat(1, 3, 1, 1)
        # img = (img - mean) / std
        # start_time = round(time.monotonic() * 1000)
        # img = img[:, (2, 1, 0), :, :]  # .contiguous()
        # img = img[:, (0, 0, 0), :, :]  # .contiguous()
        img = img.repeat(1, 3, 1, 1)
        img = (img - mean_rev) / std_rev

        dt_yolact_in = img  # .contiguous()

        # with torch.no_grad():
        #     dt_yolact_in = yolact_transform(dt_frame_yolact)

        time_taken_yolact_prepare = round(time.monotonic() * 1000) - start_time
        time.sleep(0.0050)
        start_time = round(time.monotonic() * 1000)
        with torch.no_grad():
            dets_out = yolact_net(dt_yolact_in)
            if USE_SYNC:
                torch.cuda.synchronize()
        time_taken_yolact_inference = round(time.monotonic() * 1000) - start_time
        start_time = round(time.monotonic() * 1000)
        # first_batch = [d_torch_yolact], out
        # frames = [{'value': yolact_extract_frame(first_batch, 0), 'idx': 0}]
        # dets_out = out
        # img = d_torch_yolact  # frames[0]['value'][0]
        # h, w, _ = d_torch_yolact.shape
        save = yolact_cfg.rescore_bbox
        yolact_cfg.rescore_bbox = True
        with torch.no_grad():
            t = yolact_postprocess(dets_out, TORCH_YOLACT_SHAPE[1], TORCH_YOLACT_SHAPE[0],
                                       visualize_lincomb=yolact_display_lincomb,
                                       crop_masks=yolact_crop,
                                       score_threshold=score_threshold)
        yolact_cfg.rescore_bbox = save
        idx = t[1].argsort(0, descending=True)[:top_k]
        # classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        num_dets_to_consider = min(top_k, classes.shape[0])
        # num_dets_to_consider = 0  # !!!
        for j in range(num_dets_to_consider):
            if scores[j] < score_threshold:
                num_dets_to_consider = j
                break
        if num_dets_to_consider == 0:
            mp_face_bboxes_count.value = 0
            continue
        masks = t[3][idx]
        masks = masks[:num_dets_to_consider, :, :, None]
        for j in reversed(range(num_dets_to_consider)):
            _class = classes[j]
            score = scores[j]
            x1, y1, x2, y2 = boxes[j, :]
            # np_bbox = boxes[j, :]
            xc = (x2 + x1) / 2
            yc = (y2 + y1) / 2
            ignore = False
            for k in reversed(range(num_dets_to_consider)):
                if k == j:
                    continue
                k_class = classes[k]
                k_score = scores[k]
                k_x1, k_y1, k_x2, k_y2 = boxes[k, :]
                k_xc = (k_x2 + k_x1) / 2
                k_yc = (k_y2 + k_y1) / 2
                if ((k_xc > x1) and (k_xc < x2) and (k_yc > y1) and (k_yc < y2)) or ((xc > k_x1)
                                                                                     and (
                                                                                             xc < k_x2) and (
                                                                                             yc > k_y1) and (
                                                                                             yc < k_y2)):
                    if score <= k_score:
                        ignore = True
                        break
            if ignore:
                continue
            xRel = x1 / TORCH_YOLACT_SHAPE[1]
            yRel = y1 / TORCH_YOLACT_SHAPE[0]
            wRel = (x2 - x1) / TORCH_YOLACT_SHAPE[1]
            hRel = (y2 - y1) / TORCH_YOLACT_SHAPE[0]
            # relBBox = np_bbox / yolactInputShapeArray
            # relBBoxArea = np.prod(relBBox[2:] - relBBox[:2])
            if (wRel * hRel < YOLACT_SURFACE_AREA_MIN_THRESHOLD) or (wRel * hRel > 1):
                continue
            # if (relBBoxArea < 0.0006) or (relBBoxArea > 1):
            #     continue
            xRelWide = xRel - wRel * 0.1
            yRelWide = yRel - hRel * 0.1
            wRelWide = wRel * 1.2
            hRelWide = hRel * 1.2
            # relBBoxMul09 = relBBox * 0.9
            # relBBoxMul11 = relBBox * 1.1
            # relBBoxMul01 = relBBox * 0.1
            # relBBoxWide[0] = relBBoxMul09[0] - relBBoxMul01[2]
            # relBBoxWide[1] = relBBoxMul09[1] - relBBoxMul01[3]
            # relBBoxWide[2] = relBBoxMul11[2] - relBBoxMul01[0]
            # relBBoxWide[3] = relBBoxMul11[3] - relBBoxMul01[1]
            # relBBoxWide = np.hstack((relBBoxMul09[:2], relBBoxMul11[2:])) - relBBoxMul01[[2, 3, 0, 1]]
            face_data = FaceData()
            face_data.bbox_relative = np.clip(
                np.array([xRelWide, yRelWide, wRelWide, hRelWide], dtype=np.float32), 0, 1)
            face_data.bbox_relativeOriginal = xRel, yRel, wRel, hRel
            np_face_bboxes[len(frame_faces_list)] = np.array([[xRel, yRel, wRel, hRel], [xRelWide, yRelWide, wRelWide, hRelWide]], dtype=np.float32)
            if wRel == 0 or hRel == 0:
                wRel = wRel
            # np_face_bboxes[j, 1] = np.array([xRel, yRel, wRel, hRel], dtype=np.float32)
            wide_x1 = int(xRelWide * TORCH_YOLACT_SHAPE[1])
            wide_y1 = int(yRelWide * TORCH_YOLACT_SHAPE[0])
            wide_x2 = int(wRelWide * TORCH_YOLACT_SHAPE[1] + wide_x1)
            wide_y2 = int(hRelWide * TORCH_YOLACT_SHAPE[0] + wide_y1)
            if _class == 0:
                face_data.face_mask = 1
            else:
                face_data.face_mask = -1
            mask_gpu = masks[j]
            mask_roi_gpu = masks[j, y1:y2, x1:x2]
            mask_roi_wide_gpu = masks[j, wide_y1:wide_y2, wide_x1:wide_x2]
            nonzeros = torch.nonzero(mask_roi_gpu.data, as_tuple=False).size(0)
            mask_roi_np = None
            mask_roi_wide_np = None
            mask_roi_wide_resized_np = None
            if nonzeros > mask_roi_gpu.shape[0] * mask_roi_gpu.shape[1] / 4:
                # mask = np.zeros((mask_gpu.shape[0], mask_gpu.shape[1], 1), dtype=np.float)
                # mask_roi_np = mask_roi_gpu.cpu().numpy()
                # mask_roi_wide_np = mask_roi_wide_gpu.cpu().numpy()
                # mask_roi_wide_resized_np = None  # cv2.resize(mask_roi_wide_gpu.cpu().numpy(), (UEM_MASK_FACE_WIDTH, UEM_MASK_FACE_HEIGHT))
                # mask[y1:y2, x1:x2] = mask_roi_np
                pass
            else:
                mask = None
            # face_data.shape_mask = mask
            # face_data.shape_roi_mask = mask_roi_np
            # face_data.shape_roi_mask_wide = mask_roi_wide_np
            # face_data.shape_roi_mask_wide_resized = mask_roi_wide_resized_np
            frame_faces_list.append(face_data)
        mp_face_bboxes_count.value = len(frame_faces_list)
        time_taken_yolact_process = round(time.monotonic() * 1000) - start_time

        if mp_face_bboxes_count.value == 0:
            continue
        time.sleep(0.0050)
        # Prepare face tensors
        start_time = round(time.monotonic() * 1000)
        dt_frame_yolact = torch.as_tensor(cp_frame_placeholder[..., :3], device='cuda')
        dt_frame_yolact = dt_frame_yolact.unsqueeze(0)
        dt_frame_yolact = dt_frame_yolact.float()
        dt_faces_128 = None
        dt_faces_160 = None
        for face_data_idx in range(mp_face_bboxes_count.value):
            # xRelWide, yRelWide, wRelWide, hRelWide = face_data.bbox_relative
            # x, y, w, h = (np.array(face_data.bbox_relative) * np.array(
            #     [cap_width, cap_height, cap_width, cap_height])).astype(np.int32)
            x, y, w, h = (np_face_bboxes[face_data_idx, 0] * np.array(
                [FRAME_SIZE[1], FRAME_SIZE[0], FRAME_SIZE[1], FRAME_SIZE[0]])).astype(np.int32)
            if h == 0 or w == 0:
                h = h
            dt_face = dt_frame_yolact[:, y:y + h, x:x + w, :]
            dt_face = dt_face.permute(0, 3, 1, 2)
            dt_face_128 = torch.nn.functional.interpolate(dt_face, FACE_SIZE_DETECT_MASK)
            dt_face_128 /= 255
            dt_faces_128 = dt_face_128 if dt_faces_128 is None else torch.cat((dt_faces_128, dt_face_128), dim=0)
            dt_face_160 = torch.nn.functional.interpolate(dt_face, FACE_SIZE_FIND_FEATURES)
            dt_face_160 /= 255
            dt_faces_160 = dt_face_160 if dt_faces_160 is None else torch.cat((dt_faces_160, dt_face_160), dim=0)
        time_taken_prepare_face_tensors = round(time.monotonic() * 1000) - start_time
        time.sleep(0.0050)
        # Detect masks
        start_time = round(time.monotonic() * 1000)
        with torch.no_grad():
            dt_out_face_masks = uem_mask_model(dt_faces_128)
            if USE_SYNC:
                torch.cuda.synchronize()
        time_taken_masks_inference = round(time.monotonic() * 1000) - start_time
        time.sleep(0.0050)
        # Detect features
        start_time = round(time.monotonic() * 1000)
        with torch.no_grad():
            dt_out_face_features = timesler_model(dt_faces_160)
            if USE_SYNC:
                torch.cuda.synchronize()
        time_taken_face_features_inference = round(time.monotonic() * 1000) - start_time
        time.sleep(0.0050)
        # Recognize faces
        start_time = round(time.monotonic() * 1000)
        dl_out_face_features = to_dlpack(dt_out_face_features)
        cp_out_face_features = cp.fromDlpack(dl_out_face_features)
        cp_face_features_new = cp_out_face_features
        # cp_face_features_new = cp.zeros((dt_faces_128.shape[0], cp_face_features_base.shape[0]), dtype=cp.float32)
        # cp_dist_holder = cp.zeros((cp_face_features_new.shape[0], cp_face_features_base.shape[0]), dtype=cp.float32)
        for face_features_new_idx, cp_face_features_new_item in enumerate(cp_face_features_new):
            # cp_face_features_base = cp.divide(cp_face_features_base, cp.sqrt(cp.sum(cp.multiply(cp_face_features_base, cp_face_features_base), axis=1))[:, None])
            # cp_face_features_new_item = cp.divide(cp_face_features_new_item, cp.sqrt(cp.sum(cp.multiply(cp_face_features_new_item, cp_face_features_new_item), axis=1))[:, None])
            # Using prenorm (or too much GPU RAM would be required)
            cp.subtract(cp_face_features_base, cp_face_features_new_item, out=cp_diff_holder)
            cp.multiply(cp_diff_holder, cp_diff_holder, out=cp_diff_holder)
            euclidean_distances_array = cp.sum(cp_diff_holder, axis=1)
            cp_dist_holder[face_features_new_idx, :] = cp.sqrt(euclidean_distances_array)
            # cp_dist_holder[face_features_new_idx, :] = cp.linalg.norm(cp_diff_holder, axis=1)
        if USE_SYNC:
            cp.cuda.stream.get_current_stream().synchronize()
        time_taken_recognize_faces = round(time.monotonic() * 1000) - start_time

        # print("bboxes", mp_face_bboxes_count.value,
        #       "total", time_taken_yolact_prepare
        #       + time_taken_yolact_inference
        #       + time_taken_yolact_process
        #       + time_taken_prepare_face_tensors
        #       + time_taken_masks_inference
        #       + time_taken_face_features_inference
        #       + time_taken_recognize_faces,
        #       "yo_pr", time_taken_yolact_prepare,
        #       "yo_in", time_taken_yolact_inference,
        #       "yo_ps", time_taken_yolact_process,
        #       # "dr_bb", time_taken_draw_bboxes,
        #       "pr_ft", time_taken_prepare_face_tensors,
        #       "ms_in", time_taken_masks_inference,
        #       "ff_in", time_taken_face_features_inference,
        #       "rc_fc", time_taken_recognize_faces,
        #       )

def processRecognizeFacesGpu(dh_face_features_base):
    import cupy as cp
    import time

    ca_face_features_base = dh_face_features_base.open()
    cp_face_features_base = cp.asarray(ca_face_features_base)

    cp_diff_holder = cp.zeros_like(cp_face_features_base)

    cp_out_face_features = cp.zeros((5, 512), dtype=cp.float32)

    cp_dist_holder = cp.zeros((512, FACE_FEATURES_BASE_SIZE), dtype=cp.float32)

    while True:
        time.sleep(0.0100)
        # Recognize faces
        start_time = round(time.monotonic() * 1000)
        # dl_out_face_features = to_dlpack(dt_out_face_features)
        # cp_out_face_features = cp.fromDlpack(dl_out_face_features)
        cp_face_features_new = cp_out_face_features
        # cp_face_features_new = cp.zeros((dt_faces_128.shape[0], cp_face_features_base.shape[0]), dtype=cp.float32)
        for face_features_new_idx, cp_face_features_new_item in enumerate(cp_face_features_new):
            # cp_face_features_base = cp.divide(cp_face_features_base, cp.sqrt(cp.sum(cp.multiply(cp_face_features_base, cp_face_features_base), axis=1))[:, None])
            # cp_face_features_new_item = cp.divide(cp_face_features_new_item, cp.sqrt(cp.sum(cp.multiply(cp_face_features_new_item, cp_face_features_new_item), axis=1))[:, None])
            # Using prenorm (or too much GPU RAM would be required)
            cp.subtract(cp_face_features_base, cp_face_features_new_item, out=cp_diff_holder)
            cp.multiply(cp_diff_holder, cp_diff_holder, out=cp_diff_holder)
            euclidean_distances_array = cp.sum(cp_diff_holder, axis=1)
            cp_dist_holder[face_features_new_idx, :] = cp.sqrt(euclidean_distances_array)
            # cp_dist_holder[face_features_new_idx, :] = cp.linalg.norm(cp_diff_holder, axis=1)
        if USE_SYNC:
            cp.cuda.stream.get_current_stream().synchronize()
        time_taken_recognize_faces = round(time.monotonic() * 1000) - start_time

        # print(
        #       "rc_fc", time_taken_recognize_faces,
        #       )

def processRecognizeFacesCpu(dh_face_features_base):
    import cupy as cp
    import numpy as np
    import time
    import os

    print(os.sched_getaffinity(0))

    # os.system("taskset -p 0xff %d" % os.getpid())
    # os.system("taskset -p 0xfff %d" % os.getpid())
    os.system('taskset -p 0xffffffff %d' % os.getpid())
    # OPENBLAS_MAIN_FREE=1 python myscript.py

    print(os.sched_getaffinity(0))
    # os.sched_setaffinity(0, {1, 3})

    ca_face_features_base = dh_face_features_base.open()
    cp_face_features_base = cp.asarray(ca_face_features_base)

    np_face_features_base = cp.asnumpy(cp_face_features_base)

    np_diff_holder = np.zeros_like(np_face_features_base)

    np_out_face_features = np.zeros((5, 512), dtype=np.float32)

    np_dist_holder = np.zeros((512, FACE_FEATURES_BASE_SIZE), dtype=np.float32)

    np.__config__.show()

    while True:
        time.sleep(0.0100)
        # Recognize faces
        start_time = round(time.monotonic() * 1000)
        np_face_features_new = np_out_face_features
        # cp_face_features_new = cp.zeros((dt_faces_128.shape[0], cp_face_features_base.shape[0]), dtype=cp.float32)
        for face_features_new_idx, np_face_features_new_item in enumerate(np_face_features_new):
            # cp_face_features_base = cp.divide(cp_face_features_base, cp.sqrt(cp.sum(cp.multiply(cp_face_features_base, cp_face_features_base), axis=1))[:, None])
            # cp_face_features_new_item = cp.divide(cp_face_features_new_item, cp.sqrt(cp.sum(cp.multiply(cp_face_features_new_item, cp_face_features_new_item), axis=1))[:, None])
            # Using prenorm (or too much GPU RAM would be required)
            np.subtract(np_face_features_base, np_face_features_new_item, out=np_diff_holder)
            '''np.multiply(np_diff_holder, np_diff_holder, out=np_diff_holder)
            euclidean_distances_array = np.sum(np_diff_holder, axis=1)
            np_dist_holder[face_features_new_idx, :] = np.sqrt(euclidean_distances_array)'''
            # a_min_b = np.subtract(np_face_features_base, np_face_features_new_item)
            a_min_b = np_diff_holder
            np_dist_holder[face_features_new_idx, :] = np.sqrt(np.einsum("ij,ij->i", a_min_b, a_min_b))
            # cp_dist_holder[face_features_new_idx, :] = cp.linalg.norm(cp_diff_holder, axis=1)
        # if USE_SYNC:
        #     cp.cuda.stream.get_current_stream().synchronize()
        time_taken_recognize_faces = round(time.monotonic() * 1000) - start_time

        print(
              "rc_fc", time_taken_recognize_faces,
              )

def processGpuCpuTransfer(dh_face_features_base):
    import numpy as np
    import cupy as cp
    import time
    import math

    # TEST_FEATURES_COUNT = 30_000
    TEST_FEATURES_COUNT = FACE_FEATURES_BASE_SIZE

    np_arr = np.random.rand(TEST_FEATURES_COUNT, 512).astype(np.float32)

    if USE_MEMMAP:
        MEMMAP_FEATURES_COUNT = 1_000_000

        import pathlib
        SCRIPT_FOLDER_PATH = str(pathlib.Path().absolute())
        filename = SCRIPT_FOLDER_PATH + "/../memmap/memmap.npy"
        np_memmap = np.memmap(filename, dtype=np.float32, mode='r+', shape=(MEMMAP_FEATURES_COUNT, 512))
        np_arr = np_memmap[:TEST_FEATURES_COUNT, ...]
        test_step = math.ceil(MEMMAP_FEATURES_COUNT / TEST_FEATURES_COUNT)
        # features_skip = test_step - 1

    cp_arr = cp.random.rand(TEST_FEATURES_COUNT, 512).astype(np.float32)
    if USE_SYNC:
        cp.cuda.stream.get_current_stream().synchronize()

    if USE_MEMMAP:
        start_feature = 0
    while True:

        start_time = round(time.monotonic() * 1000)
        if USE_MEMMAP:
            np_mem_arr = np_memmap[start_feature::test_step, ...]
            start_feature += 1
            if start_feature >= test_step:
                start_feature = 0
            cp_arr[:np_mem_arr.shape[0], ...] = cp.asarray(np_mem_arr)
        else:
            cp_arr[...] = cp.asarray(np_arr)
        if USE_SYNC:
            cp.cuda.stream.get_current_stream().synchronize()
        time_taken_face_features_upload = round(time.monotonic() * 1000) - start_time

        start_time = round(time.monotonic() * 1000)
        np_arr[...] = cp.asnumpy(cp_arr)
        # if USE_MEMMAP:
        #     np_arr.flush()
        if USE_SYNC:
            cp.cuda.stream.get_current_stream().synchronize()
        time_taken_face_features_download = round(time.monotonic() * 1000) - start_time

        # print(
        #     "ff_ul", time_taken_face_features_upload,
        #     "ff_dl", time_taken_face_features_download,
        # )

if __name__ == '__main__':
    import sys
    import threading
    import time

    import cv2
    import numpy as np
    from pafy import pafy
    # from scipy.misc import ascent
    # https://pixnio.com/photos/people
    import OpenGL.GL as gl
    import wx
    from wx.glcanvas import GLCanvas

    # import cv2
    # print(cv2.getBuildInformation())

    import cupy as cp
    from cupy.cuda import runtime
    dev = cp.cuda.Device(runtime.getDevice())

    import pycuda
    import pycuda.driver
    import pycuda.gl

    import pycuda.autoinit

    # curr_gpu = pycuda.autoinit.device
    # ctx_gl = pycuda.gl.make_context(curr_gpu, flags=pycuda.gl.graphics_map_flags.NONE)

    # sudo systemctl restart nvargus-daemon
    #vcap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)
    #vcap = cv2.VideoCapture(("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink").format(1920, 1080), cv2.CAP_GSTREAMER)

    #vcap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)

    #vcap = cv2.VideoCapture(("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True").format(1920, 1080), cv2.CAP_GSTREAMER)
    #vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
    #vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink max-buffers=1 drop=True", cv2.CAP_GSTREAMER)
    #vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=(string)BGRx ! videorate ! videoconvert ! appsink", cv2.CAP_GSTREAMER)

    # vcap = cv2.VideoCapture("rtspsrc framerate=25 location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! rtph264depay ! h264parse ! omxh264dec enable-low-outbuffer=1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videorate ! video/x-raw, framerate=(fraction)25/1 ! videoconvert ! appsink max-buffers=1 drop=True", cv2.CAP_GSTREAMER)

    #vcap = cv2.VideoCapture("rtspsrc framerate=25 location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! rtph264depay ! h264parse ! omxh264dec enable-low-outbuffer=1 ! videorate ! video/x-raw, framerate=(fraction)25/1 ! videoconvert ! appsink max-buffers=1 drop=True", cv2.CAP_GSTREAMER)
    #vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! rtph264depay ! h264parse ! avdec_h264 max-threads=1 ! videoconvert ! appsink max-buffers=1 drop=True", cv2.CAP_GSTREAMER)
    #vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! queue max-size-buffers=1 leaky=downstream ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
    #vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! queue max-size-buffers=1 leaky=downstream ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink max-buffers=1 drop=True", cv2.CAP_GSTREAMER)
    #vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! appsink", cv2.CAP_GSTREAMER)
    #vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! queue ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080,format=(string)BGRx ! queue ! videoconvert ! queue ! appsink sync=false", cv2.CAP_GSTREAMER)
    #vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! decodebin ! appsink", cv2.CAP_GSTREAMER)
    #vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=20 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
    #vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=200 ! rtph264depay ! h264parse ! queue ! omxh264dec ! nvvidconv ! appsink", cv2.CAP_GSTREAMER)
    #vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=20 ! decodebin ! nvvidconv ! appsink", cv2.CAP_GSTREAMER)
    #vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
    #vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 ! qtdemux ! queue ! h264parse ! nvv4l2decoder ! nv3dsink -e", cv2.CAP_GSTREAMER)
    #vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=500 ! rtph264depay ! h264parse ! omxh264dec ! nvoverlaysink overlay-x=800 overlay-y=50 overlay-w=640 overlay-h=480 overlay=2", cv2.CAP_GSTREAMER)
    #vcap = cv2.VideoCapture("rtsp://admin:LABCC0805%24@192.168.7.147", cv2.CAP_FFMPEG)
    #vcap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    #vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=100 ! queue ! rtph264depay ! h264parse ! avdec_h264 max-threads=1 ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080,format=(string)BGRx ! queue ! videoconvert ! queue ! appsink sync=false", cv2.CAP_GSTREAMER)
    #vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=100 ! queue ! rtph264depay ! h264parse ! avdec_h264 max-threads=1 ! videoconvert ! queue ! appsink sync=false", cv2.CAP_GSTREAMER)
    # gst-launch-1.0 rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! decodebin ! autovideosink
    # gst-launch-1.0 rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=20 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink
    # gst-launch-1.0 rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=500 ! rtph264depay ! h264parse ! omxh264dec ! nvoverlaysink overlay-x=800 overlay-y=50 overlay-w=640 overlay-h=480 overlay=2
    # rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 ! decodebin ! nvvidconv ! video/x-raw, format=I420, width=1920, height=816 ! appsink
    # gst-launch-1.0 rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 ! rtph264depay ! queue ! h264parse ! omxh264dec ! nveglglessink -e
    # gst-launch-1.0 rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 ! qtdemux ! queue ! h264parse ! nvv4l2decoder ! nv3dsink -e
    # gst-launch-1.0 rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 ! qtdemux ! h264parse ! queue ! omxh264dec ! nvvidconv ! tee ! xvimagesink

    # gst-launch-1.0 nvarguscamerasrc sensor_id=0 ! nvoverlaysink

    cap_width, cap_height, cap_channels = FRAME_SIZE[1], FRAME_SIZE[0], 4

    vPafy = pafy.new(YOUTUBE_URL)
    # play = vPafy.getbest()  # (preftype="webm")
    play = vPafy.getbestvideo()
    print(play)
    # play = vPafy.getbest(preftype="video:mp4@1920x1080")
    # cap_width, cap_height = play.dimensions
    if USE_GPU_GRAB:
        d_reader = cv2.cudacodec.createVideoReader(play.url)
    #     while True:
    #         res, dcv_frame = d_reader.nextFrame()
    #         if res:
    #             cap_height, cap_width, cap_channels = dcv_frame.size()
    #             break
    else:
        vcap = cv2.VideoCapture(play.url)
    #     while True:
    #         ret, frame = vcap.read()
    #         if ret:
    #             cap_height, cap_width, cap_channels = frame.shape
    #             break

    '''rtsp = 'rtsp://admin:LABCC0805%24@192.168.7.147'
    rtspGstreamer = 'rtspsrc location=rtsp://admin:LABCC0805%24@192.168.7.147 latency=0 buffer-mode=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink max-buffers=1 drop=True'
    if USE_GPU_GRAB:
        d_reader = cv2.cudacodec.createVideoReader(rtsp)
    else:
        vcap = cv2.VideoCapture(rtspGstreamer, cv2.CAP_GSTREAMER)
    FRAME_SIZE = (1088, 1920)'''

    test_acc = 0
    test_cnt = 0

    # WINDOW_SIZE = (512, 512)
    WINDOW_SIZE = (cap_width, cap_height)

    class Canvas(GLCanvas):

        def __init__(self, parent):
            """create the canvas """
            super(Canvas, self).__init__(parent)
            self.textureImage = None
            self.parent = parent

            self.context = wx.glcanvas.GLContext(self)

            # execute self.onPaint whenever the parent frame is repainted
            self.Bind(wx.EVT_PAINT, self.onPaint)

            self.Bind(wx.EVT_WINDOW_DESTROY, self.onDestroy)
            self.Bind(wx.EVT_CLOSE, self.OnClose)

            '''self.parent.timerUpdateUI = wx.Timer(self.parent)
            self.parent.Bind(wx.EVT_TIMER, self.UpdateUI)
            self.parent.timerUpdateUI.Start(1000. / 1)'''

            self.closeThreads = False
            self.hGetNormalFrameThread = threading.Thread(target=self.getNormalFrameThread)  # , args=(self,))
            self.hGetNormalFrameThread.start()

            self.frame = None

            # pycuda.driver.init()
            # dev = pycuda.driver.Device(0)
            # pycuda.gl.init()
            # self.cuda_gl_context = pycuda.gl.make_context(dev)
            # pycuda.gl.BufferObjectMapping()

            self.SetCurrent(self.context)

            # gl.glEnable(gl.GL_BLEND)
            # # gl.glBlendFunc(gl.GL_ONE_MINUS_SRC_COLOR, gl. GL_ONE_MINUS_SRC_COLOR)
            # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        def initTexture(self):
            """init the texture - this has to happen after an OpenGL context
            has been created
            """

            # make the OpenGL context associated with this canvas the current one
            self.SetCurrent(self.context)

            curr_gpu = pycuda.autoinit.device
            self.ctx_gl = pycuda.gl.make_context(curr_gpu, flags=pycuda.gl.graphics_map_flags.NONE)  # WRITE_DISCARD

            #self.data = np.uint8(np.flipud(ascent()))
            #self.data = np.zeros((1080, 1920), dtype=np.uint8)
            # self.data = np.zeros((cap_height, cap_width, 4), dtype=np.uint8)
            '''img_path = '/home/thermalview/Desktop/ThermalView/saved_images/2020_11_23__13_31_03/2020_11_23__13_31_03_FULL_COLOR_SOURCE.jpg'
            self.data = cv2.imread(img_path)
            self.data = np.flipud(self.data)
            self.data = np.dstack((self.data, np.zeros(self.data.shape[:-1])))
            # self.data = np.dstack((self.data, np.zeros((self.data.shape[:-1]) + (1,))))'''

            # ret, frame = vcap.read()
            # if ret:
            #     frame = np.flipud(frame)
            #     self.data = np.dstack((frame, np.zeros(frame.shape[:-1], dtype=frame.dtype)))

            # init a buffer
            # https://stackoverflow.com/questions/21765604/draw-image-from-vertex-buffer-object-generated-with-cuda-using-opengl
            # gl.glGenBuffers(1, buffer)
            self.bufferImage = gl.glGenBuffers(1)
            # gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.buffer)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.bufferImage)
            # gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, cap_width * cap_height * 4, None, gl.GL_DYNAMIC_DRAW)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, cap_width * cap_height * 4, None, gl.GL_DYNAMIC_DRAW)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

            self.bufferText = gl.glGenBuffers(1)
            # gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.buffer)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.bufferText)
            # gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, cap_width * cap_height * 4, None, gl.GL_DYNAMIC_DRAW)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, cap_width * cap_height * 4, None, gl.GL_DYNAMIC_DRAW)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

            # this.bufferResource = new CUgraphicsResource();
            # cuGraphicsGLRegisterBuffer(bufferResource, self.buffer,
            #                            CUgraphicsMapResourceFlags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)

            # generate a texture id, make it current
            self.textureImage = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureImage)
            # texture mode and parameters controlling wrapping and scaling
            gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_MODULATE)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            # map the image data to the texture. note that if the input
            # type is GL_FLOAT, the values must be in the range [0..1]
            # gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.data.shape[1], self.data.shape[0], 0, gl.GL_LUMINANCE, gl.GL_UNSIGNED_BYTE, self.data)
            # gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, cap_width, cap_height, 0, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, self.data)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, cap_width, cap_height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

            self.textureText = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureText)
            # texture mode and parameters controlling wrapping and scaling
            gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_MODULATE)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            # map the image data to the texture. note that if the input
            # type is GL_FLOAT, the values must be in the range [0..1]
            # gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.data.shape[1], self.data.shape[0], 0, gl.GL_LUMINANCE, gl.GL_UNSIGNED_BYTE, self.data)
            # gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, cap_width, cap_height, 0, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, self.data)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, cap_width, cap_height, 0, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, None)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

            self.cuda_buf = pycuda.gl.RegisteredBuffer(int(self.bufferImage))  # , pycuda.gl.graphics_map_flags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD)
            # cuda_buf_mapping = cuda_buf.map()
            self.cuda_buf_mapping = self.cuda_buf.map()
            d_buf_ptr, d_buf_size = self.cuda_buf_mapping.device_ptr_and_size()
            # memory = cp.cuda.Memory(2048000000)
            # ptr = cp.cuda.MemoryPointer(memory, 0)
            self.cp_mem = cp.cuda.memory.UnownedMemory(d_buf_ptr, d_buf_size, self.cuda_buf_mapping)
            self.cp_memptr = cp.cuda.memory.MemoryPointer(self.cp_mem, 0)
            # memptr.memset(100, 100000)
            # cp_rand = cp.random.randint(100, size=(cap_height, cap_width, 4)).astype(cp.uint8)
            # cp_rand.data
            # memptr.copy_from_device_async(cp_rand.)
            # memptr.memset_async(100, 100000)
            self.cp_arr = cp.ndarray(shape=(cap_height, cap_width, 4), memptr=self.cp_memptr, dtype=cp.uint8)
            # cp_arr[0:100, 0:100, 0] = 200
            # ri = pycuda.gl.RegisteredImage(int(self.texture), gl.GL_TEXTURE_2D)


            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.bufferText)
            # gl.glBufferData(gl.GL_ARRAY_BUFFER, cap_height * cap_width * 4, self.frame, gl.GL_DYNAMIC_DRAW)
            buf_mem_pointer = gl.glMapBuffer(gl.GL_ARRAY_BUFFER, gl.GL_WRITE_ONLY)  # gl.GL_READ_WRITE)  # gl.GL_WRITE_ONLY)  # gl.GL_READ_WRITE)  #   #   #
            map_array = (gl.GLubyte * cap_height * cap_width * 4).from_address(buf_mem_pointer)
            # new_array = np.ctypeslib.as_array(map_array, shape=(cap_height * cap_width * 4,))  # .astype(np.uint8)
            new_array = np.ctypeslib.as_array(map_array, shape=(cap_height, cap_width, 4))  # .astype(np.uint8)
            # new_array = new_array.reshape((cap_height, cap_width, 4))
            # new_array = np.flip(new_array, 0)
            # new_array = np.ndarray(shape=(cap_height, cap_width, 4), dtype=np.uint8, buffer=buf_mem_pointer, offset=0)
            # import ctypes
            # new_array = np.ctypeslib.as_array(buf_mem_pointer, shape=None)
            # new_array = np.ndarray(shape=(cap_height, cap_width, 4), dtype=np.uint8, buffer=ctypes.c_void_p.from_address(buf_mem_pointer))
            # new_array[:50, :100, :] = 100
            # new_array = new_array.astype(np.uint8)
            self.np_arr = new_array.reshape((cap_height, cap_width, 4))
            # new_array3 = new_array[:, :, :3]  # .astype(np.uint8)
            # new_array3 = new_array3.view(np.uint8)
            # new_array3 += 128
            # new_array3 = np.ascontiguousarray(new_array, dtype=np.uint8)

            gl.glUnmapBuffer(gl.GL_ARRAY_BUFFER)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

            #self.data[30:40, 30:40] = 255

            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_DST_ALPHA)
            # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_COLOR)
            # gl.glBlendFunc(gl.GL_SRC_COLOR, gl.GL_ONE_MINUS_DST_COLOR)  # +
            # gl.glBlendFunc(gl.GL_SRC_COLOR, gl.GL_ONE_MINUS_SRC_COLOR) # +
            # gl.glBlendFunc(gl.GL_SRC_COLOR, gl.GL_ONE_MINUS_SRC1_COLOR) # +
            # gl.glBlendFunc(gl.GL_SRC_COLOR, gl.GL_ONE_MINUS_SRC_ALPHA) # +
            # gl.glBlendFunc(gl.GL_SRC_COLOR, gl.GL_ONE_MINUS_SRC1_ALPHA) # +
            # gl.glBlendFunc(gl.GL_SRC_COLOR, gl.GL_ONE_MINUS_CONSTANT_ALPHA) # +
            # gl.glBlendFunc(gl.GL_SRC_COLOR, gl.GL_ONE_MINUS_CONSTANT_COLOR) # +
            # gl.glBlendFunc(gl.GL_DST_COLOR, gl.GL_ONE_MINUS_SRC1_COLOR)
            # gl.glBlendFunc(gl.GL_DST_ALPHA, gl.GL_ONE_MINUS_SRC1_COLOR)  # ++
            # gl.glBlendFunc(gl.GL_DST_ALPHA, gl.GL_ONE_MINUS_SRC_COLOR)  # ++
            # gl.glBlendFunc(gl.GL_DST_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)  # +++
            # gl.glBlendFunc(gl.GL_DST_ALPHA, gl.GL_ONE_MINUS_SRC1_ALPHA)  # ++
            # gl.glBlendFunc(gl.GL_DST_ALPHA, gl.GL_ONE_MINUS_DST_COLOR)  #
            # gl.glBlendFuncSeparate(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA, gl.GL_ONE, gl.GL_ZERO)
            # gl.glAlphaFunc(gl.GL_NOTEQUAL, 0)

            gl.glEnable(gl.GL_TEXTURE_2D)



        def onPaint(self, event):
            """called when window is repainted """
            # make sure we have a texture to draw
            if not self.textureImage:
                self.initTexture()
                # make the OpenGL context associated with this canvas the current one
                # self.SetCurrent(self.context)
            self.onDraw()
            # if self.frame is not None:
            #     gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.frame.shape[1], self.frame.shape[0], gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, self.frame)
            #     self.SwapBuffers()

        def onDraw(self):
            """draw function """

            time_start_full = round(time.monotonic() * 1000)
            start_time = time_start_full
            if USE_GPU_GRAB:
                res, dcv_frame = d_reader.nextFrame()
                if not res or dcv_frame is None:
                    return
            else:
                res, frameNormal = vcap.read()
                if not res or frameNormal is None:
                    return
                frameNormal = cv2.cvtColor(frameNormal, cv2.COLOR_RGB2RGBA)
            # frame = np.flipud(frame)
            # self.frame = np.dstack((frame, np.zeros(frame.shape[:-1], dtype=frame.dtype)))
            time_taken_frame_capture = round(time.monotonic() * 1000) - start_time
            # print("time_taken_frame_capture", time_taken_frame_capture)

            time_start_prep = time_start_full

            # set the viewport and projection
            w, h = self.GetSize()
            gl.glViewport(0, 0, w, h)

            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            # gl.glOrtho(0, 1, 0, 1, 0, 1)
            gl.glOrtho(0, 1, 1, 0, 0, 1)

            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            time_taken_prep = round(time.monotonic() * 1000) - time_start_prep

            # ACTION!
            time_start_init_image = round(time.monotonic() * 1000)
            if USE_GPU_GRAB:
                d_frame_ptr = dcv_frame.cudaPtr()
                d_frame_shape = dcv_frame.size()[::-1] + (dcv_frame.channels(),)
                d_frame_size = d_frame_shape[0] * d_frame_shape[1] * d_frame_shape[2] * dcv_frame.elemSize1()
                mem = cp.cuda.memory.UnownedMemory(d_frame_ptr, d_frame_size, dcv_frame)
                memptr = cp.cuda.memory.MemoryPointer(mem, 0)
                # memptr.memset(100, 100000)
                # memptr.memset_async(100, 100000)
            if USE_SYNC:
                cp.cuda.stream.get_current_stream().synchronize()
            init_image_time_taken = round(time.monotonic() * 1000) - time_start_init_image

            time_start_copy_image = round(time.monotonic() * 1000)
            # self.cp_arr[0:100, 0:100, 0] = 200
            # self.cp_arr[0:200, 0:200, 0] = 200
            if USE_GPU_GRAB:
                cp_frame = cp.ndarray(shape=d_frame_shape, memptr=memptr, dtype=cp.uint8)  # , strides=(5120, 4, 1))
                self.cp_frame_placeholder[...] = cp_frame
            else:
                cp_frame = cp.asarray(frameNormal)
                self.cp_frame_placeholder[...] = cp_frame
            cap_height = cp_frame.shape[0]
            cap_width = cp_frame.shape[1]
            if USE_SYNC:
                cp.cuda.stream.get_current_stream().synchronize()
            copy_image_time_taken = round(time.monotonic() * 1000) - time_start_copy_image

            # new_array3[:50, :100, :] = 100
            # new_array[...] = cv2.flip(new_array, 0)

            time_start_draw_bboxes_gpu = round(time.monotonic() * 1000)
            for face_idx in range(self.mp_face_bboxes_count.value):
                # np_face_bbox = self.np_face_bboxes[face_idx]
                # cv2.cuda.rectangle(cp_frame, (x1, y1), (x2, y2), color, 1)
                x, y, w, h = (self.np_face_bboxes[face_idx, 0] * np.array(
                    [FRAME_SIZE[1], FRAME_SIZE[0], FRAME_SIZE[1], FRAME_SIZE[0]])).astype(np.int32)
                # cp_test = cp.full((h, 2, 3), cp.array([0, 255, 0], dtype=cp.uint8))  # , dtype=cp.uint8)
                cp_frame[y:y + h, x:x + 2, :3] = cp.array([0, 255, 0])
                cp_frame[y:y + h, x + w:x + w + 2, :3] = cp.array([0, 255, 0])
                cp_frame[y:y + 2, x:x + w, :3] = cp.array([0, 255, 0])
                cp_frame[y + h:y + h + 2, x:x + w, :3] = cp.array([0, 255, 0])
            if USE_SYNC:
                cp.cuda.stream.get_current_stream().synchronize()
            draw_bboxes_gpu_time_taken = round(time.monotonic() * 1000) - time_start_draw_bboxes_gpu

            self.cp_arr[...] = cp_frame[...]

            time_start_print_text = round(time.monotonic() * 1000)
            '''self.np_arr.fill(0)
            for face_idx in range(self.mp_face_bboxes_count.value):
                # np_face_bbox = self.np_face_bboxes[face_idx]
                x, y, w, h = (self.np_face_bboxes[face_idx, 0] * np.array(
                    [FRAME_SIZE[1], FRAME_SIZE[0], FRAME_SIZE[1], FRAME_SIZE[0]])).astype(np.int32)
                # np_text_frame = np.zeros_like(self.np_arr)
                np_text_frame = np.zeros((cap_height, cap_width, 4), dtype=np.uint8)
                text = "M! " + str(time_start_print_text)
                # text_pos = (100, 100)
                text_pos = (int(x + w + 3), int(y + 20 * (0.5 + h / 512)))
                text_size = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_TRIPLEX, 2,  # 0.4 + w / 400,  # fontScale,
                    2,  # thickness
                )
                cv2.putText(np_text_frame,
                            text,
                            text_pos,
                            cv2.FONT_HERSHEY_TRIPLEX,  # font,
                            2,  # -2,
                            (0, 255, 0, 255),
                            2,  # thickness
                            cv2.LINE_AA)

                # self.np_arr[...] = np_text_frame[...]
                self.np_arr[text_pos[1]-text_size[0][1]-1:text_pos[1]+3, text_pos[0]:text_pos[0]+text_size[0][0], :] = \
                    np_text_frame[text_pos[1]-text_size[0][1]-1:text_pos[1]+3, text_pos[0]:text_pos[0]+text_size[0][0], :]
                # new_array[...] = np.flipud(new_array)

                # self.ctx_gl.synchronize()'''
            print_text_time_taken = round(time.monotonic() * 1000) - time_start_print_text

            time_start_copy_tex = round(time.monotonic() * 1000)

            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.bufferImage)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureImage)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, cap_width, cap_height, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, None)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.bufferText)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureText)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, cap_width, cap_height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

            time_taken_tex = round(time.monotonic() * 1000) - time_start_copy_tex

            time_start_post = round(time.monotonic() * 1000)

            # enable textures, bind to our texture
            # gl.glEnable(gl.GL_TEXTURE_2D)

            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureImage)
            # gl.glColor3f(1, 1, 1)
            # draw a quad
            gl.glBegin(gl.GL_QUADS)
            gl.glTexCoord2f(0, 1)
            gl.glVertex2f(0, 1)
            gl.glTexCoord2f(0, 0)
            gl.glVertex2f(0, 0)
            gl.glTexCoord2f(1, 0)
            gl.glVertex2f(1, 0)
            gl.glTexCoord2f(1, 1)
            gl.glVertex2f(1, 1)
            gl.glEnd()
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0) #my

            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureText)
            # gl.glColor3f(1, 1, 1)
            # draw a quad
            gl.glBegin(gl.GL_QUADS)
            gl.glTexCoord2f(0, 1)
            gl.glVertex2f(0, 1)
            gl.glTexCoord2f(0, 0)
            gl.glVertex2f(0, 0)
            gl.glTexCoord2f(1, 0)
            gl.glVertex2f(1, 0)
            gl.glTexCoord2f(1, 1)
            gl.glVertex2f(1, 1)
            gl.glEnd()
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0) #my

            # gl.glDisable(gl.GL_TEXTURE_2D)

            time_taken_post = round(time.monotonic() * 1000) - time_start_post

            # gl.glFlush()

            # gl.gXSwapIntervalEXT()
            # swap the front and back buffers so that the texture is visible
            time_start_swap = round(time.monotonic() * 1000)
            self.SwapBuffers()
            time_taken_swap = round(time.monotonic() * 1000) - time_start_swap

            time_taken_full = round(time.monotonic() * 1000) - time_start_full
            global test_cnt, test_acc
            test_cnt += 1
            test_acc += time_taken_full
            print("total", time_taken_full,
                  "rtotal", round(test_acc / test_cnt),
                  "capture", time_taken_frame_capture,
                  "init_image", init_image_time_taken,
                  "copy_image", copy_image_time_taken,
                  "prep", time_taken_prep,
                  "bboxes_gpu", draw_bboxes_gpu_time_taken,
                  "text", print_text_time_taken,
                  "tex", time_taken_tex,
                  "post", time_taken_post,
                  "swap", time_taken_swap
                  )

        def getNormalFrameThread(self):
            while not self.closeThreads:
                time.sleep(0.001)
                # time.sleep(0.010)
                # time.sleep(0.100)
                self.UpdateUI(None)

        def UpdateUI(self, evt):
            # start_time_total = round(time.monotonic() * 1000)
            # start_time = start_time_total
            # res, dcv_frame = d_reader.nextFrame()
            # if not res or dcv_frame is None:
            #     return
            # # frame = np.flipud(frame)
            # # self.frame = np.dstack((frame, np.zeros(frame.shape[:-1], dtype=frame.dtype)))
            # self.dcv_frame = dcv_frame
            # time_taken_frame_capture = round(time.monotonic() * 1000) - start_time
            # print("time_taken_frame_capture", time_taken_frame_capture)
            self.parent.Refresh()
            # wx.CallAfter(self.parent.Refresh)

        # def UpdateUI(self, evt):
        #     # ret, frame = vcap.read()
        #     # ret, frame = True, np.zeros((1080, 1920, 4), dtype=np.uint8)
        #     ret, frame = True, (np.random.rand(cap_height, cap_width, 3) * 255).astype('B')
        #     if ret:
        #         #self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #         frame = np.flipud(frame)
        #         self.frame = np.dstack((frame, np.zeros(frame.shape[:-1], dtype=frame.dtype)))
        #         #self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2BGRA)
        #         wx.CallAfter(self.parent.Refresh)

        def OnClose(self, event):
            self.Destroy()

        def onDestroy(self, event=None):
            ## clean up resources as needed here
            del self.cp_arr
            self.cp_arr = None
            del self.cp_memptr
            self.cp_memptr = None
            del self.cp_mem
            self.cp_mem = None
            self.cuda_buf_mapping.unmap()
            del self.cuda_buf_mapping
            self.cuda_buf_mapping = None
            self.cuda_buf.unregister()
            del self.cuda_buf
            self.cuda_buf = None
            event.Skip()

    def main():
        import multiprocessing as mp
        from multiprocessing import Process

        mp.set_start_method('spawn')

        import numba.cuda
        import pycuda
        pycuda.driver.init()
        pycuda.driver.Device(0).retain_primary_context()

        cp_frame_placeholder = cp.zeros(FRAME_SIZE + (FRAME_CHANNELS,), dtype=cp.uint8)
        ca_frame_placeholder = numba.cuda.as_cuda_array(cp_frame_placeholder)
        dh_frame_placeholder = ca_frame_placeholder.get_ipc_handle()

        mp_face_bboxes_count = mp.Value('B', 0)
        # np_face_bboxes = np.zeros((256, 2, 4), dtype=np.float32)
        mp_face_bboxes = mp.Array('f', FACE_BBOXES_SHAPE[0] * FACE_BBOXES_SHAPE[1] * FACE_BBOXES_SHAPE[2])

        np_face_bboxes = np.frombuffer(mp_face_bboxes.get_obj(), dtype=np.float32)
        np_face_bboxes = np_face_bboxes.reshape(FACE_BBOXES_SHAPE)

        cp_face_features_base = cp.zeros((FACE_FEATURES_BASE_SIZE, FACE_FEATURES_COUNT), dtype=cp.float32)
        ca_face_features_base = numba.cuda.as_cuda_array(cp_face_features_base)
        dh_face_features_base = ca_face_features_base.get_ipc_handle()

        procGpuPipeline = Process(target=processGpuPipeline, args=(dh_frame_placeholder,
                                                                   mp_face_bboxes_count, mp_face_bboxes,
                                                                   dh_face_features_base))  #, dh_render_placeholder))
        procGpuPipeline.start()

        procRecognizeFacesGpu = Process(target=processRecognizeFacesGpu, args=(dh_face_features_base,))
        # procRecognizeFacesGpu.start()

        procRecognizeFacesCpu = Process(target=processRecognizeFacesCpu, args=(dh_face_features_base,))
        # procRecognizeFacesCpu.start()

        procGpuCpuTransfer = Process(target=processGpuCpuTransfer, args=(dh_face_features_base,))
        # procGpuCpuTransfer.start()

        app = wx.App()
        fr = wx.Frame(None, size=WINDOW_SIZE, title='wxPython texture demo')
        canv = Canvas(fr)
        canv.cp_frame_placeholder = cp_frame_placeholder
        canv.np_face_bboxes = np_face_bboxes
        canv.mp_face_bboxes_count = mp_face_bboxes_count
        fr.Show()
        app.MainLoop()

        procGpuCpuTransfer.join()
        procRecognizeFacesCpu.join()
        procRecognizeFacesGpu.join()
        procGpuPipeline.join()

    main()
