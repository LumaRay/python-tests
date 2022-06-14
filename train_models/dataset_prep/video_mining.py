import pafy, cv2, numpy as np

USE_TORCH_YOLACT = True
USE_TF_FACED = True

if USE_TORCH_YOLACT:
    # https://github.com/dbolya/yolact
    from face_detection.yolact.yolact import Yolact
    import torch
    import torch.backends.cudnn as cudnn
    from face_detection.yolact.utils.functions import SavePath as yolact_SavePath
    from face_detection.yolact.config import cfg as yolact_cfg, set_cfg as yolact_set_cfg, COLORS as yolact_COLORS
    from face_detection.yolact.utils.augmentations import FastBaseTransform as yolact_FastBaseTransform
    # from face_detection.yolact.utils.functions import MovingAverage
    from face_detection.yolact.layers.output_utils import postprocess as yolact_postprocess, undo_image_transformation as yolact_undo_image_transformation
    from collections import defaultdict

    yolact_color_cache = defaultdict(lambda: {})
    #yolact_trained_model = '../face_detection/yolact/weights/yolact_base_54_800000.pth'
    #yolact_trained_model = 'f:\\Work\\InfraredCamera\\ThermalView\\face_detection\\yolact\\weights\\yolact_resnet50_54_800000.pth'
    yolact_trained_model = '../face_detection/yolact/weights/yolact_resnet50_54_800000.pth'
    #yolact_trained_model = '../face_detection/yolact/weights/yolact_heads_3399_34000.pth'
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

        # evaluate(net, dataset)
        # def evaluate(net:Yolact, dataset, train_mode=False):
        yolact_net.detect.use_fast_nms = yolact_fast_nms
        yolact_net.detect.use_cross_class_nms = yolact_cross_class_nms
        yolact_cfg.mask_proto_debug = yolact_mask_proto_debug

        # evalvideo(net, args.video)
        # def evalvideo(net:Yolact, path:str, out_path:str=None):

        class yolact_CustomDataParallel(torch.nn.DataParallel):
            """ A Custom Data Parallel class that properly gathers lists of dictionaries. """

            def gather(self, outputs, output_device):
                # Note that I don't actually want to convert everything to the output_device
                return sum(outputs, [])

        yolact_net = yolact_CustomDataParallel(yolact_net).cuda()
        yolact_transform = torch.nn.DataParallel(yolact_FastBaseTransform()).cuda()
        yolact_extract_frame = lambda x, i: (
            x[0][i] if x[1][i]['detection'] is None else x[0][i].to(x[1][i]['detection']['box'].device), [x[1][i]])

if USE_TF_FACED:
    from face_detection.faced_new import FaceDetector as faced_FaceDetector
    faced_face_detector = faced_FaceDetector()

url = 'https://www.youtube.com/watch?v=xn7wPPSh6yI'
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
        out = yolact_net(imgs)
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
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape
        save = yolact_cfg.rescore_bbox
        yolact_cfg.rescore_bbox = True
        t = yolact_postprocess(dets_out, w, h, visualize_lincomb=yolact_display_lincomb,
                               crop_masks=yolact_crop,
                               score_threshold=yolact_score_threshold)
        yolact_cfg.rescore_bbox = save
        idx = t[1].argsort(0, descending=True)[:yolact_top_k]
        if yolact_cfg.eval_mask_branch:
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
        if yolact_display_masks and yolact_cfg.eval_mask_branch and num_dets_to_consider > 0:
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
            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        img_numpy = (img_gpu * 255).byte().cpu().numpy()
        if num_dets_to_consider > 0:
            if yolact_display_text or yolact_display_bboxes:
                for j in reversed(range(num_dets_to_consider)):
                    x1, y1, x2, y2 = boxes[j, :]
                    if (classes[j] == 0) and ((x2 - x1) > 50):
                        y_min = int(max(0, y1 - (x2 - x1) / 2))
                        y_max = int((y1 + y2) / 1.5)
                        x_min = int(max(0, x1 - (x2 - x1) / 1))
                        x_max = int(min(frame.shape[1], x2 + (x2 - x1) / 1))
                        sub_frame = frame[y_min:y_max, x_min:x_max]
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
                            cv2.rectangle(sub_frame, (rx_min, ry_min), (rx_max, ry2), (0, 255, 0), 1)
                        cv2.imshow('frame1', sub_frame)
                    color = get_color(j)
                    score = scores[j]
                    if yolact_display_bboxes:
                        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
                    if yolact_display_text:
                        _class = yolact_cfg.dataset.class_names[classes[j]]
                        text_str = '%s: %.2f' % (_class, score) if yolact_display_scores else _class
                        font_face = cv2.FONT_HERSHEY_DUPLEX
                        font_scale = 0.6
                        font_thickness = 1
                        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                        text_pt = (x1, y1 - 3)
                        text_color = [255, 255, 255]
                        cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                        cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                                    cv2.LINE_AA)

    frame = img_numpy








    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

