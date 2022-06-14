import pafy, cv2, numpy as np

USE_TORCH_YOLACT = True
USE_TF_FACED = True

HEAD_SIZE_MIN = 30

if USE_TORCH_YOLACT:
    # https://github.com/dbolya/yolact
    from face_detection.yolact.yolact_heads import Yolact as heads_Yolact
    from face_detection.yolact.yolact_people import Yolact as people_Yolact
    import torch
    import torch.backends.cudnn as cudnn
    from face_detection.yolact.utils.functions import SavePath as yolact_SavePath
    #from face_detection.yolact.config import COLORS as yolact_COLORS
    from face_detection.yolact.config_heads import cfg_heads as yolact_cfg_heads, set_cfg_heads as yolact_set_cfg_heads, COLORS as yolact_COLORS
    from face_detection.yolact.config_people import cfg_people as yolact_cfg_people, set_cfg_people as yolact_set_cfg_people
    from face_detection.yolact.utils.augmentations import FastBaseTransform as yolact_FastBaseTransform
    # from face_detection.yolact.utils.functions import MovingAverage
    #from face_detection.yolact.layers.output_utils import undo_image_transformation as yolact_undo_image_transformation
    from face_detection.yolact.layers.output_utils_heads import postprocess as yolact_postprocess_heads, undo_image_transformation as yolact_undo_image_transformation
    from face_detection.yolact.layers.output_utils_people import postprocess as yolact_postprocess_people
    from collections import defaultdict

    yolact_color_cache = defaultdict(lambda: {})
    #yolact_trained_model = '../face_detection/yolact/weights/yolact_base_54_800000.pth'
    #yolact_trained_model = 'f:\\Work\\InfraredCamera\\ThermalView\\face_detection\\yolact\\weights\\yolact_resnet50_54_800000.pth'
    yolact_trained_model_people = '../face_detection/yolact/weights/yolact_resnet50_54_800000.pth'
    yolact_trained_model_heads = '../face_detection/yolact/weights/yolact_heads_3399_34000.pth'
    #yolact_trained_model = pathToScriptFolder + '/face_detection/yolact/weights/yolact_maskfaces_5117_51173_interrupt.pth'
    #yolact_trained_model = pathToScriptFolder + '/face_detection/yolact/weights/yolact_maskfaces_10952_109522_interrupt.pth'
    #yolact_trained_model = pathToScriptFolder + '/face_detection/yolact/weights/yolact_darknet53_54_800000.pth'
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
    yolact_score_threshold = 0.15
    yolact_crop = True

    yolact_model_path_heads = yolact_SavePath.from_str(yolact_trained_model_heads)
    yolact_config_heads = yolact_model_path_heads.model_name + '_config'
    yolact_set_cfg_heads(yolact_config_heads)

    yolact_model_path_people = yolact_SavePath.from_str(yolact_trained_model_people)
    yolact_config_people = yolact_model_path_people.model_name + '_config'
    yolact_set_cfg_people(yolact_config_people)

    with torch.no_grad():
        if yolact_cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        yolact_net_heads = heads_Yolact()
        yolact_net_heads.load_weights(yolact_trained_model_heads)
        yolact_net_heads.eval()

        if yolact_cuda:
            yolact_net_heads = yolact_net_heads.cuda()

        yolact_net_people = people_Yolact()
        yolact_net_people.load_weights(yolact_trained_model_people)
        yolact_net_people.eval()

        if yolact_cuda:
            yolact_net_people = yolact_net_people.cuda()

        # evaluate(net, dataset)
        # def evaluate(net:Yolact, dataset, train_mode=False):
        yolact_net_people.detect.use_fast_nms = yolact_fast_nms
        yolact_net_people.detect.use_cross_class_nms = yolact_cross_class_nms
        yolact_net_heads.detect.use_fast_nms = yolact_fast_nms
        yolact_net_heads.detect.use_cross_class_nms = yolact_cross_class_nms
        yolact_cfg_people.mask_proto_debug = yolact_mask_proto_debug
        yolact_cfg_heads.mask_proto_debug = yolact_mask_proto_debug

        # evalvideo(net, args.video)
        # def evalvideo(net:Yolact, path:str, out_path:str=None):

        class yolact_CustomDataParallel(torch.nn.DataParallel):
            """ A Custom Data Parallel class that properly gathers lists of dictionaries. """

            def gather(self, outputs, output_device):
                # Note that I don't actually want to convert everything to the output_device
                return sum(outputs, [])

        yolact_net_heads = yolact_CustomDataParallel(yolact_net_heads).cuda()
        yolact_net_people = yolact_CustomDataParallel(yolact_net_people).cuda()
        yolact_transform = torch.nn.DataParallel(yolact_FastBaseTransform()).cuda()
        yolact_extract_frame = lambda x, i: (x[0][i] if x[1][i]['detection'] is None else x[0][i].to(x[1][i]['detection']['box'].device), [x[1][i]])

#url = 'https://www.youtube.com/watch?v=xn7wPPSh6yI'
#url = 'https://www.youtube.com/watch?v=HC9FDfuUpKQ'
#url = 'https://www.youtube.com/watch?v=B-giWTnAYPw'
#url = 'https://www.youtube.com/watch?v=FuIh710CUao'
#url = 'https://www.youtube.com/watch?v=RsrpV-GweKc' # +++
#url = 'https://www.youtube.com/watch?v=BE31Y46s-uQ' # +++
url = 'https://www.youtube.com/watch?v=_wfSCvsaaos'
#url = 'https://www.youtube.com/watch?v=VOCNWyy3zJU' # ---

#url = 'https://youtu.be/SxIUyECUEik'
#url = 'https://youtu.be/3LwWl2wU4tQ'
#url = 'https://youtu.be/uSLZfNteDxM'
vPafy = pafy.new(url)
play = vPafy.getbest()

cap = cv2.VideoCapture(play.url)

i = 0
while (True):
    ret, frame = cap.read()

    if not ret:
        break

    i += 1

    if i % 4 != 0:
        continue

    head_boxes_mask = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)

    head_bboxes = []

    img = frame
    with torch.no_grad():
        # first_batch = eval_network(transform_frame(get_next_frame(vid)))
        frame_width = img.shape[1]
        frame_height = img.shape[0]
        frames = [img.copy()]
        frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
        imgs = yolact_transform(torch.stack(frames, 0))
        num_extra = 0
        while imgs.size(0) < yolact_video_multiframe:
            imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
            num_extra += 1
        out = yolact_net_heads(imgs)
        if num_extra > 0:
            out = out[:-num_extra]
        first_batch = frames, out
        frames = [{'value': yolact_extract_frame(first_batch, 0), 'idx': 0}]
        # def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
        dets_out = out
        # img = frames
        img = frames[0]['value'][0]
        h = frame_height
        w = frame_width
        # undo_transform = True
        undo_transform = False
        class_color = False
        mask_alpha = 0.45
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
        if undo_transform:
            img_numpy = yolact_undo_image_transformation(img, w, h)
            # img_numpy = rgb_imgNormal
            img_gpu_heads = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu_heads = img / 255.0
            h, w, _ = img.shape
        save = yolact_cfg_heads.rescore_bbox
        yolact_cfg_heads.rescore_bbox = True
        t = yolact_postprocess_heads(dets_out, w, h, visualize_lincomb=yolact_display_lincomb,
                               crop_masks=yolact_crop,
                               score_threshold=yolact_score_threshold)
        yolact_cfg_heads.rescore_bbox = save
        idx = t[1].argsort(0, descending=True)[:yolact_top_k]
        if yolact_cfg_heads.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        num_dets_to_consider = min(yolact_top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < yolact_score_threshold:
                num_dets_to_consider = j
                break


        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed
        def get_color(j, on_gpu=None):
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(yolact_COLORS)

            if on_gpu is not None and color_idx in yolact_color_cache[on_gpu]:
                return yolact_color_cache[on_gpu][color_idx]
            else:
                color = yolact_COLORS[color_idx]
                if not undo_transform:
                    # The image might come in as RGB or BRG, depending
                    color = (color[2], color[1], color[0])
                if on_gpu is not None:
                    color = torch.Tensor(color).to(on_gpu).float() / 255.
                    yolact_color_cache[on_gpu][color_idx] = color
                return color


        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        '''if yolact_display_masks and yolact_cfg_heads.eval_mask_branch and num_dets_to_consider > 0:
            # After this, mask is of size [num_dets, h, w, 1]
            masks = masks[:num_dets_to_consider, :, :, None]

            # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
            colors = torch.cat(
                [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in
                 range(num_dets_to_consider)],
                dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
            # This is 1 everywhere except for 1-mask_alpha where the mask is
            inv_alph_masks = masks * (-mask_alpha) + 1
            # I did the math for this on pen and paper. This whole block should be equivalent to:
            #    for j in range(num_dets_to_consider):
            #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
            masks_color_summand = masks_color[0]
            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)
            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand'''
        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        img_numpy_heads = (img_gpu_heads * 255).byte().cpu().numpy()
        if num_dets_to_consider > 0:
            if yolact_display_text or yolact_display_bboxes:
                for j in reversed(range(num_dets_to_consider)):
                    x1, y1, x2, y2 = boxes[j, :]
                    if (classes[j] == 0) and ((x2 - x1) > HEAD_SIZE_MIN):
                        if y1 > 0:
                            h = y2 - y1
                            w = x2 - x1
                            h = max(h, w * 4 / 3)
                            y2 = int(y1 + h)
                        y_min = int(max(0, y1 - (x2 - x1) / 1))
                        #y_max = int((y1 + y2) / 1.5)
                        y_max = y2
                        x_min = int(max(0, x1 - (x2 - x1) / 0.7))
                        x_max = int(min(frame.shape[1], x2 + (x2 - x1) / 0.7))
                        '''sub_frame = frame[y_min:y_max, x_min:x_max]
                        sub_frame = cv2.resize(sub_frame, (sub_frame.shape[1] * 3, sub_frame.shape[0] * 3))
                        #gb = cv2.GaussianBlur(sub_frame, (0, 0), 3)
                        #sub_frame = cv2.addWeighted(sub_frame, 1.5, gb, -0.5, 0)
                        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        sub_frame = cv2.filter2D(sub_frame, -1, kernel)
                        bboxesCompressed = faced_face_detector.predict(sub_frame, 0.8)
                        for xc, yc, w, h, p in bboxesCompressed:
                            rx1 = int(xc - w / 2)
                            ry1 = int(yc - h / 2)
                            rx2 = int(xc + w / 2)
                            ry2 = int(yc + h / 2)
                            rx_min = int(max(0, rx1 - (rx2 - rx1) / 2))
                            ry_min = int(max(0, ry1 - (rx2 - rx1) / 2))
                            rx_max = int(min(frame.shape[1], rx2 + (rx2 - rx1) / 2))
                            cv2.rectangle(sub_frame, (rx_min, ry_min), (rx_max, ry2), (0, 255, 0), 1)'''
                        #cv2.imshow('frame1', sub_frame)
                        cv2.rectangle(img_numpy_heads, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.rectangle(head_boxes_mask, (x_min, y_min), (x_max, y_max), (255, 255, 255), cv2.FILLED)
                        #head_bboxes.append((x_min, y_min, x_max, y_max))
                        head_bboxes.append((x1, y1, x2, y2))

                    color = get_color(j)
                    score = scores[j]
                    '''if yolact_display_bboxes:
                        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
                    if yolact_display_text:
                        _class = yolact_cfg_heads.dataset.class_names[classes[j]]
                        text_str = '%s: %.2f' % (_class, score) if yolact_display_scores else _class
                        font_face = cv2.FONT_HERSHEY_DUPLEX
                        font_scale = 0.6
                        font_thickness = 1
                        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                        text_pt = (x1, y1 - 3)
                        text_color = [255, 255, 255]
                        cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                        cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                                    cv2.LINE_AA)'''

    #cv2.imshow('heads', img_numpy_heads)

    #cv2.imshow('head_boxes_mask', head_boxes_mask)

    #img_numpy_heads = cv2.cvtColor(img_numpy_heads, cv2.COLOR_BGR2RGB)
    #img_numpy_heads = cv2.bitwise_not(img_numpy_heads)
    #img_gpu_heads = torch.Tensor(img_numpy_heads).cuda() / 255.0















    #people_mask = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
    people_mask_gpu = torch.zeros(frame.shape[0], frame.shape[1], 3).cuda()

    img = frame
    with torch.no_grad():
        # first_batch = eval_network(transform_frame(get_next_frame(vid)))
        frame_width = img.shape[1]
        frame_height = img.shape[0]
        frames = [img.copy()]
        frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
        imgs = yolact_transform(torch.stack(frames, 0))
        num_extra = 0
        while imgs.size(0) < yolact_video_multiframe:
            imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
            num_extra += 1
        out = yolact_net_people(imgs)
        if num_extra > 0:
            out = out[:-num_extra]
        first_batch = frames, out
        frames = [{'value': yolact_extract_frame(first_batch, 0), 'idx': 0}]
        # def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
        dets_out = out
        # img = frames
        img = frames[0]['value'][0]
        h = frame_height
        w = frame_width
        # undo_transform = True
        undo_transform = False
        class_color = False
        mask_alpha = 0.45
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
        if undo_transform:
            img_numpy2 = yolact_undo_image_transformation(img, w, h)
            # img_numpy = rgb_imgNormal
            img_gpu_people = torch.Tensor(2).cuda()
        else:
            img_gpu_people = img / 255.0
            #img_gpu_people = people_mask_gpu / 255.0
            h, w, _ = img.shape
        save = yolact_cfg_people.rescore_bbox
        yolact_cfg_people.rescore_bbox = True
        t = yolact_postprocess_people(dets_out, w, h, visualize_lincomb=yolact_display_lincomb,
                               crop_masks=yolact_crop,
                               score_threshold=yolact_score_threshold)
        yolact_cfg_people.rescore_bbox = save
        idx = t[1].argsort(0, descending=True)[:yolact_top_k]
        if yolact_cfg_people.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        num_dets_to_consider = min(yolact_top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < yolact_score_threshold:
                num_dets_to_consider = j
                break


        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed
        def get_color(j, on_gpu=None):
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(yolact_COLORS)

            if on_gpu is not None and color_idx in yolact_color_cache[on_gpu]:
                return yolact_color_cache[on_gpu][color_idx]
            else:
                color = yolact_COLORS[color_idx]
                if not undo_transform:
                    # The image might come in as RGB or BRG, depending
                    color = (color[2], color[1], color[0])
                if on_gpu is not None:
                    color = torch.Tensor(color).to(on_gpu).float() / 255.
                    yolact_color_cache[on_gpu][color_idx] = color
                return color


        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        if yolact_display_masks and yolact_cfg_people.eval_mask_branch and num_dets_to_consider > 0:
            # After this, mask is of size [num_dets, h, w, 1]
            masks = masks[:num_dets_to_consider, :, :, None]

            # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
            colors = torch.cat(
                [get_color(j, on_gpu=img_gpu_people.device.index).view(1, 1, 1, 3) for j in
                 range(num_dets_to_consider)],
                dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
            # This is 1 everywhere except for 1-mask_alpha where the mask is
            inv_alph_masks = masks * (-mask_alpha) + 1
            # I did the math for this on pen and paper. This whole block should be equivalent to:
            #    for j in range(num_dets_to_consider):
            #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
            masks_color_summand = masks_color[0]
            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)
            img_gpu_people = img_gpu_people * inv_alph_masks.prod(dim=0) + masks_color_summand

            #masks_cpu = (masks * 255).byte().cpu().numpy()
        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        #img_numpy_people = (img_gpu_people * 255).byte().cpu().numpy()
        img_numpy_people = (img_gpu_people * 255).byte().cpu().numpy()
        if num_dets_to_consider > 0:
            if yolact_display_text or yolact_display_bboxes:
                for j in reversed(range(num_dets_to_consider)):
                    x1, y1, x2, y2 = boxes[j, :]
                    if (classes[j] == 0) and ((x2 - x1) > HEAD_SIZE_MIN * 1.5):
                        y_min = int(max(0, y1 - (x2 - x1) / 2))
                        y_max = int((y1 + y2) / 1.5)
                        x_min = int(max(0, x1 - (x2 - x1) / 1))
                        x_max = int(min(frame.shape[1], x2 + (x2 - x1) / 1))
                        sub_frame = frame[y_min:y_max, x_min:x_max]
                        sub_frame = cv2.resize(sub_frame, (sub_frame.shape[1] * 3, sub_frame.shape[0] * 3))
                        '''#gb = cv2.GaussianBlur(sub_frame, (0, 0), 3)
                        #sub_frame = cv2.addWeighted(sub_frame, 1.5, gb, -0.5, 0)
                        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        sub_frame = cv2.filter2D(sub_frame, -1, kernel)
                        bboxesCompressed = faced_face_detector.predict(sub_frame, 0.8)
                        for xc, yc, w, h, p in bboxesCompressed:
                            rx1 = int(xc - w / 2)
                            ry1 = int(yc - h / 2)
                            rx2 = int(xc + w / 2)
                            ry2 = int(yc + h / 2)
                            rx_min = int(max(0, rx1 - (rx2 - rx1) / 2))
                            ry_min = int(max(0, ry1 - (rx2 - rx1) / 2))
                            rx_max = int(min(frame.shape[1], rx2 + (rx2 - rx1) / 2))
                            cv2.rectangle(sub_frame, (rx_min, ry_min), (rx_max, ry2), (0, 255, 0), 1)'''
                        #cv2.imshow('frame1', sub_frame)
                        mask_cpu = (masks[j] * 255).byte().cpu().numpy()
                        #if not mask_cpu.any():
                        #if np.count_nonzero(mask_cpu) > 100:
                        for head_bbox in head_bboxes:
                            head_x1, head_y1, head_x2, head_y2 = head_bbox
                            head_center_x = int((head_x2 + head_x1) / 2)
                            head_center_y = int((head_y2 + head_y1) / 2)
                            cv2.rectangle(img_numpy_people, (head_x1, head_y1), (head_x2, head_y2), (0, 255, 0), 2)
                            head_y_min = int(max(0, head_y1 - (head_x2 - head_x1) / 1))
                            # head_y_max = int((y1 + y2) / 1.5)
                            head_y_max = head_y2
                            head_x_min = int(max(0, head_x1 - (head_x2 - head_x1) / 0.7))
                            head_x_max = int(min(frame.shape[1], head_x2 + (head_x2 - head_x1) / 0.7))
                            cv2.rectangle(img_numpy_people, (head_x_min, head_y_min), (head_x_max, head_y_max), (255, 0, 0), 2)
                            if (x1 < head_center_x) and (head_center_x < x2) and (y1 < head_center_y) and (head_center_y < y2):
                                cv2.imshow('masks', mask_cpu[head_y_min:head_y_max, head_x_min:head_x_max])
                            #cv2.imshow('masks', mask_cpu * (head_boxes_mask / 255))
                    color = get_color(j)
                    score = scores[j]
                    if yolact_display_bboxes:
                        #cv2.rectangle(img_numpy2, (x1, y1), (x2, y2), color, 1)
                        cv2.rectangle(img_numpy_people, (x1, y1), (x2, y2), (255, 255, 0), 1)
                    '''if yolact_display_text:
                        _class = yolact_cfg_people.dataset.class_names[classes[j]]
                        text_str = '%s: %.2f' % (_class, score) if yolact_display_scores else _class
                        font_face = cv2.FONT_HERSHEY_DUPLEX
                        font_scale = 0.6
                        font_thickness = 1
                        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                        text_pt = (x1, y1 - 3)
                        text_color = [255, 255, 255]
                        cv2.rectangle(img_numpy2, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                        cv2.putText(img_numpy2, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                                    cv2.LINE_AA)'''

    cv2.imshow('people', img_numpy_people)

    #cv2.imshow('frame', frame)

    #cv2.imshow('res', img_numpy2 * (head_boxes_mask / 255))





    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

