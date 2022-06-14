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

if __name__ == '__main__':
    def main():
        import time
        import numpy as np
        import cupy as cp
        import cv2
        import multiprocessing as mp
        from multiprocessing import Process
        from pafy import pafy

        mp.set_start_method('spawn')

        vPafy = pafy.new("https://www.youtube.com/watch?v=HC9FDfuUpKQ")
        play = vPafy.getbest()  # (preftype="webm")
        d_reader = cv2.cudacodec.createVideoReader(play.url)

        while True:
            time.sleep(0.040)
            res, dcv_frame = d_reader.nextFrame()
            if not res:
                start_time = round(time.monotonic() * 1000)
                continue
            cp_frame = gpuMatToCuPy(cp, dcv_frame)
            cp_frame_placeholder[...] = cp_frame[...]
            cp_render_placeholder[...] = cp_frame[...]
            # Download frame to cpu
            start_time = round(time.monotonic() * 1000)
            np_frame = dcv_frame.download()
            time_taken_frame_download = round(time.monotonic() * 1000) - start_time

            # Draw text on cpu
            start_time = round(time.monotonic() * 1000)
            for face_data in frame_faces_list:
                xRelWide, yRelWide, wRelWide, hRelWide = face_data.bbox_relative
                x, y, w, h = (np.array(face_data.bbox_relative) * np.array([cap_width, cap_height, cap_width, cap_height])).astype(
                    np.int32)
                fontColor = (88, 255, 133)
                if True:
                    fontColor = (255, 88, 133)
                text = "M!"
                text_size = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_TRIPLEX, 0.4 + w / 400,  # fontScale,
                    1,  # thickness
                )
                cv2.putText(np_frame,
                            text,
                            (0, text_size[0][1] - 1),
                            # (x + w + 3, int(y + 20 * (0.5 + h / 512))),  # position,
                            cv2.FONT_HERSHEY_TRIPLEX,  # font,
                            0.4 + w / 400,  # fontScale,
                            fontColor,
                            1,  # thickness
                            cv2.LINE_AA)
            time_taken_draw_text_cpu = round(time.monotonic() * 1000) - start_time

            # Show frame on cpu
            # cv2.cuda.imshow("test", d_frame)
            start_time = round(time.monotonic() * 1000)
            cv2.imshow("test", np_frame)
            time_taken_frame_show = round(time.monotonic() * 1000) - start_time

            time_taken_total = round(time.monotonic() * 1000) - start_time_total
            # cv2.imshow("test", np_frame)
            cv2.waitKey(1)

            print("total", time_taken_total,
                  "fr_ca", time_taken_frame_capture,
                  "cv_cp", time_taken_cv_to_cp,
                  "yo_pr", time_taken_yolact_prepare,
                  "yo_in", time_taken_yolact_inference,
                  "yo_ps", time_taken_yolact_process,
                  "dr_bb", time_taken_draw_bboxes,
                  "pr_ft", time_taken_prepare_face_tensors,
                  "ms_in", time_taken_masks_inference,
                  "ff_in", time_taken_face_features_inference,
                  "rc_fc", time_taken_recognize_faces,
                  "dr_tx_d", time_taken_draw_text_cpu_gpu,
                  "fr_dl", time_taken_frame_download,
                  "dr_tx_p", time_taken_draw_text_cpu,
                  "fr_sh", time_taken_frame_show,
                  )

    main()

