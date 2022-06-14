import time

FRAME_SIZE = (720, 1280)
FRAME_CHANNELS = 4
FACE_BBOXES_SHAPE = (256, 2, 4)
FACE_SIZE_DETECT_MASK = (128, 128)
FACE_SIZE_FIND_FEATURES = (160, 160)


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

def gpuMatToCuPy(cp, dcv_frame):
    d_frame_ptr = dcv_frame.cudaPtr()
    d_frame_shape = dcv_frame.size()[::-1] + (dcv_frame.channels(),)
    # cap_height = d_frame_shape[0]
    # cap_width = d_frame_shape[1]
    d_frame_size = d_frame_shape[0] * d_frame_shape[1] * d_frame_shape[2] * dcv_frame.elemSize1()
    mem = cp.cuda.memory.UnownedMemory(d_frame_ptr, d_frame_size, dcv_frame)
    memptr = cp.cuda.memory.MemoryPointer(mem, 0)
    # # memptr.memset(100, 100000)
    # # memptr.memset_async(100, 100000)
    cp_frame = cp.ndarray(shape=d_frame_shape, memptr=memptr, dtype=cp.uint8)  # , strides=(5120, 4, 1))
    return cp_frame

def processCaptureFrames(dh_frame_placeholder, dh_render_placeholder):
    import cv2
    from pafy import pafy
    import numpy as np
    import cupy as cp
    import time

    vPafy = pafy.new("https://www.youtube.com/watch?v=HC9FDfuUpKQ")
    play = vPafy.getbest()  # (preftype="webm")
    d_reader = cv2.cudacodec.createVideoReader(play.url)
    # capNormal = cv2.VideoCapture(play.url)
    # with yt_frame_arr.get_lock():
    #     yt_frame = np.frombuffer(yt_frame_arr.get_obj(), dtype=np.uint8)
    #     yt_frame = yt_frame.reshape((720, 1280, 3))
    start_time = round(time.monotonic() * 1000)
    # dcv_first_frame = None
    ca_frame_placeholder = dh_frame_placeholder.open()
    cp_frame_placeholder = cp.asarray(ca_frame_placeholder)
    ca_render_placeholder = dh_render_placeholder.open()
    cp_render_placeholder = cp.asarray(ca_render_placeholder)
    while True:
        time.sleep(0.040)
        # time.sleep(0.050)
        # time.sleep(0.500)
        time_taken = round(time.monotonic() * 1000) - start_time
        # print("processCaptureYoutube loop idle", str(time_taken), "ms") # 10 ms
        start_time = round(time.monotonic() * 1000)
        # ret, frameNormal = capNormal.read()
        res, dcv_frame = d_reader.nextFrame()
        # print("processCaptureYoutube loop", str(time_taken), "ms")  # , str(mp_dcv_frame_ptr.value))  # yt_dlp 1 ms
        if not res:
            start_time = round(time.monotonic() * 1000)
            continue
        cp_frame = gpuMatToCuPy(cp, dcv_frame)
        cp_frame_placeholder[...] = cp_frame[...]
        cp_render_placeholder[...] = cp_frame[...]
        # else:
        #     cuda.cudaMemcpy(dcv_first_frame.cudaPtr(), dcv_frame.cudaPtr(), FRAME_SIZE[0] * FRAME_SIZE[1] * FRAME_CHANNELS, cuda.cudaMemcpyDeviceToDevice)
        time_taken = round(time.monotonic() * 1000) - start_time
        # cv2.imshow("123", frameNormal)
        # cv2.waitKey(1)
        start_time = round(time.monotonic() * 1000)
    cp_frame_placeholder = None
    ca_frame_placeholder = None
    dh_frame_placeholder.close()
    cp_render_placeholder = None
    ca_render_placeholder = None
    dh_render_placeholder.close()

def processDetectFaces(dh_frame_placeholder, mp_face_bboxes_count, mp_face_bboxes):
    TORCH_YOLACT_SHAPE = (640, 640)

    import numpy as np
    import cupy as cp

    from face_detection.yolact.yolact import Yolact
    import torch
    from torch.utils.dlpack import to_dlpack
    import torch.backends.cudnn as cudnn
    from face_detection.yolact.utils.functions import SavePath as yolact_SavePath
    from face_detection.yolact.config import cfg as yolact_cfg, set_cfg as yolact_set_cfg, COLORS as yolact_COLORS
    from face_detection.yolact.utils.augmentations import FastBaseTransform as yolact_FastBaseTransform
    from face_detection.yolact.layers.output_utils import postprocess as yolact_postprocess

    yolact_trained_model = '/home/thermalview/Desktop/ThermalView/face_detection/yolact/weights/yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbgray640_64_146000.pth'  # n845/349 849/354 852/353 499:851/353 854/347 (830/388 833/383 831/393 828/391)

    yolact_fast_nms = True
    yolact_cross_class_nms = False
    yolact_display_lincomb = False
    yolact_mask_proto_debug = False
    yolact_crop = True

    score_threshold = 7 / 100  # self.settings.propFaceDetectionTorchYolactThreshold
    top_k = 150  # self.settings.propFaceDetectionTorchYolactTopK

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

    ca_frame_placeholder = dh_frame_placeholder.open()
    cp_frame_placeholder = cp.asarray(ca_frame_placeholder)

    np_face_bboxes = np.frombuffer(mp_face_bboxes.get_obj(), dtype=np.float32)
    np_face_bboxes = np_face_bboxes.reshape(FACE_BBOXES_SHAPE)

    def threadDetectFaces():
        while True:
            time.sleep(0.00050)
            cp_frame_placeholder_copy = cp_frame_placeholder.copy()
            time.sleep(0.00050)
            start_time = round(time.monotonic() * 1000)
            cp_frame_gray = cp.dot(cp_frame_placeholder_copy[..., :3],
                                   cp.asarray([0.2989, 0.5870, 0.1140], dtype=cp.float32)).astype(cp.uint8)
            cp_frame_gray = cp.repeat(cp_frame_gray[:, :, cp.newaxis], 3, axis=2)

            dlp_frame = cp_frame_gray.toDlpack()
            dt_frame = torch.utils.dlpack.from_dlpack(dlp_frame)

            dt_frame_yolact = dt_frame
            # np_frame = d_torch_yolact.cpu().numpy()
            dt_frame_yolact = dt_frame_yolact.unsqueeze(0)
            # d_torch_src = d_torch_src.permute(0, 3, 1, 2)
            # d_torch_yolact = torch.nn.functional.interpolate(d_torch_yolact, TORCH_YOLACT_SHAPE)
            # np_frame = d_torch_yolact.cpu().numpy().squeeze(0)
            dt_frame_yolact = dt_frame_yolact.float()
            # d_torch_yolact = d_torch_yolact / 255

            # np_frame = d_torch_yolact.cpu().numpy().squeeze(0)

            frame_faces_list = []
            frame_width = dt_frame_yolact.shape[1]
            frame_height = dt_frame_yolact.shape[0]
            num_extra = 0
            # d_torch_yolact = yolact_transform(torch.stack(d_torch_yolact, 0))
            with torch.no_grad():
                dt_yolact_in = yolact_transform(dt_frame_yolact)
            time_taken_yolact_prepare = round(time.monotonic() * 1000) - start_time
            start_time = round(time.monotonic() * 1000)
            with torch.no_grad():
                dets_out = yolact_net(dt_yolact_in)
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
            if num_dets_to_consider > 0:
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
                    if (wRel * hRel < 0.0006) or (wRel * hRel > 1):
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
                    np_face_bboxes[j] = np.array([[xRel, yRel, wRel, hRel], [xRelWide, yRelWide, wRelWide, hRelWide]], dtype=np.float32)
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

                print("yo_pr", time_taken_yolact_prepare,
                      "yo_in", time_taken_yolact_inference,
                      "yo_ps", time_taken_yolact_process)

    import threading
    th = threading.Thread(target=threadDetectFaces)  # , args=(None,))
    th.start()
    th.join()

    cp_frame_placeholder = None
    ca_frame_placeholder = None
    dh_frame_placeholder.close()

def processDrawBBoxes(dh_render_placeholder, mp_face_bboxes_count, mp_face_bboxes):
    import cupy as cp
    import numpy as np
    ca_render_placeholder = dh_render_placeholder.open()
    cp_render_placeholder = cp.asarray(ca_render_placeholder)
    np_face_bboxes = np.frombuffer(mp_face_bboxes.get_obj(), dtype=np.float32)
    np_face_bboxes = np_face_bboxes.reshape(FACE_BBOXES_SHAPE)
    while True:
        time.sleep(0.0040)
        start_time = round(time.monotonic() * 1000)
        for face_data_idx in range(mp_face_bboxes_count.value):
            # xRelWide, yRelWide, wRelWide, hRelWide = face_data.bbox_relative
            # xRelWide, yRelWide, wRelWide, hRelWide = np_face_bboxes[face_data_idx, 1]
            # x, y, w, h = (np.array(face_data.bbox_relative) * np.array(
            #     [cap_width, cap_height, cap_width, cap_height])).astype(np.int32)
            x, y, w, h = (np_face_bboxes[face_data_idx, 0] * np.array(
                [FRAME_SIZE[1], FRAME_SIZE[0], FRAME_SIZE[1], FRAME_SIZE[0]])).astype(np.int32)
            # cp_test = cp.full((h, 2, 3), cp.array([0, 255, 0], dtype=cp.uint8))  # , dtype=cp.uint8)
            cp_render_placeholder[y:y + h, x:x + 2, :3] = cp.array([0, 255, 0])
            cp_render_placeholder[y:y + h, x + w:x + w + 2, :3] = cp.array([0, 255, 0])
            cp_render_placeholder[y:y + 2, x:x + w, :3] = cp.array([0, 255, 0])
            cp_render_placeholder[y + h:y + h + 2, x:x + w, :3] = cp.array([0, 255, 0])
        time_taken_draw_bboxes = round(time.monotonic() * 1000) - start_time
    cp_render_placeholder = None
    ca_render_placeholder = None
    dh_render_placeholder.close()

def processDetectMasks(dh_frame_placeholder, mp_face_bboxes_count, mp_face_bboxes):
    import cupy as cp
    import numpy as np
    import torch
    import imp
    with torch.no_grad():
        uem_mask_torch_weights = "/home/thermalview/Desktop/ThermalView/tests/keras_torch/keras_to_torch_uem_mask_2.pt"  # path_to_the_numpy_weights
        A = imp.load_source('MainModel',
                            '/home/thermalview/Desktop/ThermalView/tests/keras_torch/keras_to_torch_uem_mask_2.py')

        uem_mask_model = torch.load(uem_mask_torch_weights)
        # model = torch.load(torch_weights, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        uem_mask_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        uem_mask_model = uem_mask_model.to(uem_mask_device)
        uem_mask_model.eval()
    ca_frame_placeholder = dh_frame_placeholder.open()
    cp_frame_placeholder = cp.asarray(ca_frame_placeholder)
    np_face_bboxes = np.frombuffer(mp_face_bboxes.get_obj(), dtype=np.float32)
    np_face_bboxes = np_face_bboxes.reshape(FACE_BBOXES_SHAPE)
    while True:
        time.sleep(0.0040)
        if mp_face_bboxes_count.value == 0:
            continue
        # Prepare face tensors
        start_time = round(time.monotonic() * 1000)
        dt_frame_yolact = torch.as_tensor(cp_frame_placeholder[..., :3], device='cuda')
        dt_frame_yolact = dt_frame_yolact.unsqueeze(0)
        dt_frame_yolact = dt_frame_yolact.float()
        dt_faces_128 = None
        for face_data_idx in range(mp_face_bboxes_count.value):
            # xRelWide, yRelWide, wRelWide, hRelWide = face_data.bbox_relative
            # x, y, w, h = (np.array(face_data.bbox_relative) * np.array(
            #     [cap_width, cap_height, cap_width, cap_height])).astype(np.int32)
            x, y, w, h = (np_face_bboxes[face_data_idx, 0] * np.array(
                [FRAME_SIZE[1], FRAME_SIZE[0], FRAME_SIZE[1], FRAME_SIZE[0]])).astype(np.int32)
            dt_face = dt_frame_yolact[:, y:y + h, x:x + w, :]
            dt_face = dt_face.permute(0, 3, 1, 2)
            dt_face_128 = torch.nn.functional.interpolate(dt_face, FACE_SIZE_DETECT_MASK)
            dt_face_128 /= 255
            dt_faces_128 = dt_face_128 if dt_faces_128 is None else torch.cat((dt_faces_128, dt_face_128), dim=0)
        time_taken_prepare_face_tensors = round(time.monotonic() * 1000) - start_time
        # Detect masks
        start_time = round(time.monotonic() * 1000)
        with torch.no_grad():
            dt_out_face_masks = uem_mask_model(dt_faces_128)
        time_taken_masks_inference = round(time.monotonic() * 1000) - start_time

def processDetectFaceFeatures(dh_frame_placeholder, mp_face_bboxes_count, mp_face_bboxes):
    import cupy as cp
    import numpy as np
    import torch
    with torch.no_grad():
        from facenet_pytorch import InceptionResnetV1 as TimeslerInceptionResnetV1
        timesler_model = TimeslerInceptionResnetV1(pretrained='vggface2')
        timesler_model = timesler_model.eval()
        timesler_model.cuda()
    ca_frame_placeholder = dh_frame_placeholder.open()
    cp_frame_placeholder = cp.asarray(ca_frame_placeholder)
    np_face_bboxes = np.frombuffer(mp_face_bboxes.get_obj(), dtype=np.float32)
    np_face_bboxes = np_face_bboxes.reshape(FACE_BBOXES_SHAPE)
    while True:
        time.sleep(0.0040)
        if mp_face_bboxes_count.value == 0:
            continue
        # Prepare face tensors
        start_time = round(time.monotonic() * 1000)
        dt_frame_yolact = torch.as_tensor(cp_frame_placeholder[..., :3], device='cuda')
        dt_frame_yolact = dt_frame_yolact.unsqueeze(0)
        dt_frame_yolact = dt_frame_yolact.float()
        dt_faces_160 = None
        for face_data_idx in range(mp_face_bboxes_count.value):
            # xRelWide, yRelWide, wRelWide, hRelWide = face_data.bbox_relative
            # x, y, w, h = (np.array(face_data.bbox_relative) * np.array(
            #     [cap_width, cap_height, cap_width, cap_height])).astype(np.int32)
            x, y, w, h = (np_face_bboxes[face_data_idx, 0] * np.array(
                [FRAME_SIZE[1], FRAME_SIZE[0], FRAME_SIZE[1], FRAME_SIZE[0]])).astype(np.int32)
            dt_face = dt_frame_yolact[:, y:y + h, x:x + w, :]
            dt_face = dt_face.permute(0, 3, 1, 2)
            dt_face_160 = torch.nn.functional.interpolate(dt_face, FACE_SIZE_FIND_FEATURES)
            dt_face_160 /= 255
            dt_faces_160 = dt_face_160 if dt_faces_160 is None else torch.cat((dt_faces_160, dt_face_160), dim=0)
        time_taken_prepare_face_tensors = round(time.monotonic() * 1000) - start_time
        # Detect features
        start_time = round(time.monotonic() * 1000)
        with torch.no_grad():
            dt_out_face_features = timesler_model(dt_faces_160)
        time_taken_face_features_inference = round(time.monotonic() * 1000) - start_time
    cp_frame_placeholder = None
    ca_frame_placeholder.close()

def processDownloadFrames(dh_render_placeholder, mp_render_placeholder):
    import cupy as cp
    import numpy as np
    import time

    ca_render_placeholder = dh_render_placeholder.open()
    cp_render_placeholder = cp.asarray(ca_render_placeholder)

    np_render_placeholder = np.frombuffer(mp_render_placeholder.get_obj(), dtype=np.uint8)
    np_render_placeholder = np_render_placeholder.reshape(FRAME_SIZE + (FRAME_CHANNELS,))

    while True:
        time.sleep(0.040)
        start_time = round(time.monotonic() * 1000)
        np_render_placeholder[...] = cp.asnumpy(cp_render_placeholder)
        time_taken_frame_download = round(time.monotonic() * 1000) - start_time

    cp_render_placeholder = None
    ca_render_placeholder.close()

def processRenderText(mp_render_placeholder, mp_face_bboxes_count, mp_face_bboxes):
    import cv2
    import numpy as np
    import time

    np_render_placeholder = np.frombuffer(mp_render_placeholder.get_obj(), dtype=np.uint8)
    np_render_placeholder = np_render_placeholder.reshape(FRAME_SIZE + (FRAME_CHANNELS,))

    np_face_bboxes = np.frombuffer(mp_face_bboxes.get_obj(), dtype=np.float32)
    np_face_bboxes = np_face_bboxes.reshape(FACE_BBOXES_SHAPE)

    while True:
        time.sleep(0.040)
        if mp_face_bboxes_count.value == 0:
            continue
        start_time = round(time.monotonic() * 1000)
        for face_data_idx in range(mp_face_bboxes_count.value):
            # xRelWide, yRelWide, wRelWide, hRelWide = face_data.bbox_relative
            # x, y, w, h = (np.array(face_data.bbox_relative) * np.array(
            #     [cap_width, cap_height, cap_width, cap_height])).astype(np.int32)
            x, y, w, h = (np_face_bboxes[face_data_idx, 0] * np.array(
                [FRAME_SIZE[1], FRAME_SIZE[0], FRAME_SIZE[1], FRAME_SIZE[0]])).astype(np.int32)
            fontColor = (88, 255, 133)
            if True:
                fontColor = (255, 88, 133)
            text = "M!"
            text_size = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_TRIPLEX, 0.4 + w / 400,  # fontScale,
                1,  # thickness
            )
            cv2.putText(np_render_placeholder,
                        text,
                        # (0, text_size[0][1] - 1),
                        (x + w + 3, int(y + 20 * (0.5 + h / 512))),  # position,
                        cv2.FONT_HERSHEY_TRIPLEX,  # font,
                        0.4 + w / 400,  # fontScale,
                        fontColor,
                        1,  # thickness
                        cv2.LINE_AA)
        time_taken_draw_text_cpu = round(time.monotonic() * 1000) - start_time

def processShowFrames(mp_render_placeholder):
    import cv2
    import numpy as np
    import time

    np_render_placeholder = np.frombuffer(mp_render_placeholder.get_obj(), dtype=np.uint8)
    np_render_placeholder = np_render_placeholder.reshape(FRAME_SIZE + (FRAME_CHANNELS,))
    while True:
        time.sleep(0.040)
        cv2.imshow("test", np_render_placeholder)
        cv2.waitKey(1)

if __name__ == '__main__':
    def main():
        import multiprocessing as mp
        from multiprocessing import Process

        mp.set_start_method('spawn')

        # import pycuda.driver as cuda
        '''from cupy import cuda
        import pycuda
    
        cp_frame_placeholder = cuda.alloc(FRAME_SIZE[0] * FRAME_SIZE[1] * FRAME_CHANNELS)
    
        cp_frame_placeholder_ptr = cp_frame_placeholder.ptr
    
        mp_dcv_frame_ptr_arr = mp.Array('B', 8)
        # mp_dcv_frame_ptr = mp.Value('q', 1)
    
        pycuda.driver.mem_get_ipc_handle()'''

        import numba.cuda
        import cupy as cp
        import numpy as np

        cp_frame_placeholder = cp.zeros(FRAME_SIZE + (FRAME_CHANNELS,), dtype=cp.uint8)
        ca_frame_placeholder = numba.cuda.as_cuda_array(cp_frame_placeholder)
        dh_frame_placeholder = ca_frame_placeholder.get_ipc_handle()

        cp_render_placeholder = cp.zeros_like(cp_frame_placeholder)
        ca_render_placeholder = numba.cuda.as_cuda_array(cp_render_placeholder)
        dh_render_placeholder = ca_render_placeholder.get_ipc_handle()

        mp_face_bboxes_count = mp.Value('B', 0)
        # np_face_bboxes = np.zeros((256, 2, 4), dtype=np.float32)
        mp_face_bboxes = mp.Array('f', FACE_BBOXES_SHAPE[0] * FACE_BBOXES_SHAPE[1] * FACE_BBOXES_SHAPE[2])

        # np_render_placeholder = np.zeros(FRAME_SIZE + (FRAME_CHANNELS,), dtype=np.uint8)
        mp_render_placeholder = mp.Array('B', FRAME_SIZE[0] * FRAME_SIZE[1] * FRAME_CHANNELS)

        # import pickle
        # import io
        # f = io.BytesIO()
        # pickle.dump(h, f, protocol=None, fix_imports=True)
        # res = f.getvalue()
        # np_bytes = np.fromstring(res, dtype=np.uint8, sep='')
        # test_res = np_bytes.tostring()
        # if res == test_res:
        #     dsd = 5
        # r = numba.cuda.open_ipc_array(h, shape=(1,), dtype=cp.uint8, strides=(1,), offset=0)
        #
        # h_new.open()
        #
        # c_p = r.as_cupy()
        #
        # numba.cuda.M
        # procCaptureFrames = Process(target=processCaptureFrames, args=(cp_frame_placeholder_ptr, mp_dcv_frame_ptr_arr))  # cp_frame_placeholder_ptr,))
        # procShowFrames = Process(target=processShowFrames, args=(cp_frame_placeholder_ptr, mp_dcv_frame_ptr_arr,))
        procCaptureFrames = Process(target=processCaptureFrames, args=(dh_frame_placeholder, dh_render_placeholder))
        procDetectFaces = Process(target=processDetectFaces, args=(dh_frame_placeholder, mp_face_bboxes_count, mp_face_bboxes))
        procDrawBBoxes = Process(target=processDrawBBoxes, args=(dh_render_placeholder, mp_face_bboxes_count, mp_face_bboxes))
        procDetectMasks = Process(target=processDetectMasks, args=(dh_frame_placeholder, mp_face_bboxes_count, mp_face_bboxes))
        procDetectFaceFeatures = Process(target=processDetectFaceFeatures, args=(dh_frame_placeholder, mp_face_bboxes_count, mp_face_bboxes))
        procDownloadFrames = Process(target=processDownloadFrames, args=(dh_render_placeholder, mp_render_placeholder))
        procRenderText = Process(target=processRenderText, args=(mp_render_placeholder, mp_face_bboxes_count, mp_face_bboxes))
        procShowFrames = Process(target=processShowFrames, args=(mp_render_placeholder,))

        procCaptureFrames.start()
        procDetectFaces.start()
        procDrawBBoxes.start()
        procDetectMasks.start()
        procDetectFaceFeatures.start()
        procDownloadFrames.start()
        procRenderText.start()
        procShowFrames.start()

        procShowFrames.join()
        procRenderText.join()
        procDownloadFrames.join()
        procDetectFaceFeatures.join()
        procDetectMasks.join()
        procDrawBBoxes.join()
        procDetectFaces.join()
        procCaptureFrames.join()

    main()

