import time

FRAME_SIZE = (720, 1280)
FRAME_CHANNELS = 4

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

def processCaptureFrames(dh_frame_placeholder, cp_frame_placeholder_ptr, mp_dcv_frame_ptr_arr):
    import cv2
    from pafy import pafy
    import numpy as np
    import cupy as cp
    import time

    # from cupy import cuda
    # import pycuda.cuda as cuda

    # cp_frame_placeholder = cuda.alloc(FRAME_SIZE[0] * FRAME_SIZE[1] * FRAME_CHANNELS)
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
    while True:
        time.sleep(0.040)
        # time.sleep(0.050)
        # time.sleep(0.500)
        time_taken = round(time.monotonic() * 1000) - start_time
        # print("processCaptureYoutube loop idle", str(time_taken), "ms") # 10 ms
        start_time = round(time.monotonic() * 1000)
        # ret, frameNormal = capNormal.read()
        res, dcv_frame = d_reader.nextFrame()
        # with yt_frame_arr.get_lock():
        # yt_frame[...] = frameNormal[...]
        # print("processCaptureYoutube loop", str(time_taken), "ms")  # , str(mp_dcv_frame_ptr.value))  # yt_dlp 1 ms
        if not res:
            start_time = round(time.monotonic() * 1000)
            continue
        # if dcv_first_frame is None:
        #     dcv_first_frame = dcv_frame
        #     # mp_dcv_frame_ptr.value = dcv_frame.cudaPtr()
        #     # dcv_frame_ptr = np.frombuffer(mp_dcv_frame_ptr_arr.get_obj(), dtype=np.uint64)
        #     # dcv_frame_ptr[0] = dcv_first_frame.cudaPtr()
        cp_frame = gpuMatToCuPy(cp, dcv_frame)
        cp_frame_placeholder[...] = cp_frame[...]
        # else:
        #     cuda.cudaMemcpy(dcv_first_frame.cudaPtr(), dcv_frame.cudaPtr(), FRAME_SIZE[0] * FRAME_SIZE[1] * FRAME_CHANNELS, cuda.cudaMemcpyDeviceToDevice)
        time_taken = round(time.monotonic() * 1000) - start_time
        # cv2.imshow("123", frameNormal)
        # cv2.waitKey(1)
        start_time = round(time.monotonic() * 1000)
    cp_frame_placeholder = None
    ca_frame_placeholder = None
    dh_frame_placeholder.close()

def processShowFrames(dh_frame_placeholder, cp_frame_placeholder_ptr, mp_dcv_frame_ptr_arr):
    import cv2
    import numpy as np
    import cupy as cp
    import time

    ca_frame_placeholder = dh_frame_placeholder.open()
    cp_frame_placeholder = cp.asarray(ca_frame_placeholder)
    while True:
        time.sleep(0.040)
        # dcv_frame_ptr = np.frombuffer(mp_dcv_frame_ptr_arr.get_obj(), dtype=np.uint64)
        # d_frame_ptr = int(dcv_frame_ptr[0])
        # d_frame_ptr = cp_frame_placeholder_ptr
        # if d_frame_ptr == 0:
        #     continue
        # d_frame_shape = FRAME_SIZE + (FRAME_CHANNELS,)
        # cap_height = d_frame_shape[0]
        # cap_width = d_frame_shape[1]
        # o: object = object()
        # d_frame_size = d_frame_shape[0] * d_frame_shape[1] * d_frame_shape[2]
        # mem = cp.cuda.memory.UnownedMemory(d_frame_ptr, d_frame_size, o)
        # memptr = cp.cuda.memory.MemoryPointer(mem, 0)
        # memptr.memset(100, 100000)
        # memptr.memset_async(100, 100000)
        # cp_frame = cp.ndarray(shape=d_frame_shape, memptr=memptr, dtype=cp.uint8)  # , strides=(5120, 4, 1))
        np_frame = cp.asnumpy(cp_frame_placeholder)
        # np_frame = dcv_frame.download()
        cv2.imshow("test", np_frame)
        cv2.waitKey(1)
    cp_frame_placeholder = None
    ca_frame_placeholder = None
    dh_frame_placeholder.close()

def processDetectFaces(dh_frame_placeholder):
    TORCH_YOLACT_SHAPE = (640, 640)

    import numpy as np
    import cupy as cp

    from face_detection.yolact.yolact import Yolact
    import torch
    import torchvision
    from torch.utils.dlpack import to_dlpack
    import torch.backends.cudnn as cudnn
    from face_detection.yolact.utils.functions import SavePath as yolact_SavePath
    from face_detection.yolact.config import cfg as yolact_cfg, set_cfg as yolact_set_cfg, COLORS as yolact_COLORS
    from face_detection.yolact.utils.augmentations import FastBaseTransform as yolact_FastBaseTransform
    from collections import defaultdict
    from face_detection.yolact.layers.output_utils import postprocess as yolact_postprocess, \
        undo_image_transformation as yolact_undo_image_transformation

    yolact_color_cache = defaultdict(lambda: {})
    yolact_trained_model = '/home/thermalview/Desktop/ThermalView/face_detection/yolact/weights/yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbgray640_64_146000.pth'  # n845/349 849/354 852/353 499:851/353 854/347 (830/388 833/383 831/393 828/391)

    yolact_cuda = True
    yolact_top_k = 150
    yolact_fast_nms = True
    yolact_cross_class_nms = False
    yolact_display_masks = True
    yolact_display_bboxes = True
    yolact_display_text = True
    yolact_display_scores = True
    yolact_display_lincomb = False
    yolact_mask_proto_debug = False
    yolact_video_multiframe = 1
    yolact_score_threshold = 0.015
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
    while True:
        time.sleep(0.00050)
        start_time = round(time.monotonic() * 1000)
        cp_frame_placeholder_copy = cp_frame_placeholder.copy()
        cp_frame_gray = cp.dot(cp_frame_placeholder_copy[..., :3],
                               cp.asarray([0.2989, 0.5870, 0.1140], dtype=cp.float32)).astype(cp.uint8)
        cp_frame_gray = cp.repeat(cp_frame_gray[:, :, cp.newaxis], 3, axis=2)

        dlp_frame = cp_frame_gray.toDlpack()
        dt_frame = torch.utils.dlpack.from_dlpack(dlp_frame)

        # storage = dt_frame.storage()
        # device, handle, size, offset, view_size = storage._share_cuda_()

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
            time_taken_yolact_process = round(time.monotonic() * 1000) - start_time

            print("yo_pr", time_taken_yolact_prepare,
                  "yo_in", time_taken_yolact_inference,
                  "yo_ps", time_taken_yolact_process,)
        pass
    cp_frame_placeholder = None
    ca_frame_placeholder = None
    dh_frame_placeholder.close()

if __name__ == '__main__':
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

    cp_frame_placeholder = cp.zeros(FRAME_SIZE + (FRAME_CHANNELS,), dtype=cp.uint8)
    ca_frame_placeholder = numba.cuda.as_cuda_array(cp_frame_placeholder)
    dh_frame_placeholder = ca_frame_placeholder.get_ipc_handle()

    # import pickle
    # import io
    # import numpy as np
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
    procCaptureFrames = Process(target=processCaptureFrames, args=(dh_frame_placeholder, None, None))  # cp_frame_placeholder_ptr,))
    procDetectFaces = Process(target=processDetectFaces, args=(dh_frame_placeholder,))
    procShowFrames = Process(target=processShowFrames, args=(dh_frame_placeholder, None, None))

    procCaptureFrames.start()
    procDetectFaces.start()
    procShowFrames.start()

    procShowFrames.join()
    procDetectFaces.join()
    procCaptureFrames.join()

