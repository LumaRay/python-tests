# https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb

import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


display_min = 1000
display_max = 10000

def display(image, display_min, display_max):  # copied from Bi Rico
    # Here I set copy=True in order to ensure the original image is not
    # modified. If you don't mind modifying the original image, you can
    # set copy=False or skip this step.
    image = np.array(image, copy=True)
    image.clip(display_min, display_max, out=image)
    image -= display_min
    np.floor_divide(image, (display_max - display_min + 1) / 256, out=image, casting='unsafe')
    return image.astype(np.uint8)

def lut_display(image, display_min, display_max):
    lut = np.arange(2 ** 16, dtype='uint16')
    lut = display(lut, display_min, display_max)
    return np.take(lut, image)


def main():
    #if not os.path.exists(args.directory):
    #    os.mkdir(args.directory)
    try:
        config = rs.config()
        #rs.config.enable_device_from_file(config, args.input)
        pipeline = rs.pipeline()
        #config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        pipeline.start(config)

        pc = rs.pointcloud()

        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())

            width = depth_frame.get_width()
            height = depth_frame.get_height()
            dist_to_center = depth_frame.get_distance(int(width / 2), int(height / 2))
            print(dist_to_center)


            # Generate the pointcloud and texture mappings
            points = pc.calculate(depth_frame)
            points_data = np.asanyarray(points.get_data())
            color_frame = frames.get_color_frame()
            color_data = np.asanyarray(color_frame.get_data())
            # Tell pointcloud object to map to this color frame
            pc.map_to(color_frame)


            # display(image, display_min, display_max)
            img_scaled = lut_display(depth_image, display_min, display_max)


            # img_scaled = cv2.normalize(depth_image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
            # img_scaled = cv2.equalizeHist(img_scaled)

            cv2.imshow("test", img_scaled)
            #plt.imshow(depth_image, cmap="gray", vmin=0, vmax=4096)
            key = cv2.waitKey(4)
            if key == 27:  # esc
                break
    finally:
        pass


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("-d", "--directory", type=str, help="Path to save the images")
    #parser.add_argument("-i", "--input", type=str, help="Bag file to read")
    #args = parser.parse_args()

    # main()

    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1920, 1080)
    cfg.enable_stream(rs.stream.depth, 1280, 720)

    pipe = rs.pipeline()
    #decimate = rs.decimation_filter(8)
    decimate = rs.decimation_filter(4)
    #decimate = rs.decimation_filter(2)
    spatiate = rs.spatial_filter()
    spatiate.set_option(rs.option.holes_fill, 3)
    temporate = rs.temporal_filter()
    holate = rs.hole_filling_filter()
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    color_map = rs.colorizer()

    align_to = rs.stream.color
    align = rs.align(align_to)

    profile = pipe.start(cfg)

    def image_click(event, x, y, flags, param):
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            dist = depth_frame.get_distance(x, y)
            print(dist)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", image_click)

    while(True):
        # Get frameset of color and depth
        frames = pipe.wait_for_frames()
        decimated = frames
        #decimated = decimate.process(decimated).as_frameset()
        #decimated = spatiate.process(decimated).as_frameset()
        #decimated = temporate.process(decimated).as_frameset()
        decimated = decimate.process(decimated)
        decimated = depth_to_disparity.process(decimated)
        decimated = spatiate.process(decimated)
        decimated = temporate.process(decimated)
        decimated = disparity_to_depth.process(decimated)
        decimated = holate.process(decimated)
        decimated = decimated.as_frameset()

        # Align the depth frame to color frame
        aligned_frames = align.process(decimated)

        color_frame = aligned_frames.get_color_frame()
        color_data = np.asanyarray(color_frame.get_data())
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        # cv2.imshow("test", color_data)

        depth_frame = aligned_frames.get_depth_frame()
        depth_frame1 = depth_frame
        '''#depth_frame = decimate.process(depth_frame)
        depth_frame1 = depth_to_disparity.process(depth_frame1)
        depth_frame1 = spatiate.process(depth_frame1)
        depth_frame1 = temporate.process(depth_frame1)
        depth_frame1 = disparity_to_depth.process(depth_frame1)
        depth_frame1 = holate.process(depth_frame1)'''
        # depth_image = np.asanyarray(depth_frame.get_data())
        colorized_depth = np.asanyarray(color_map.colorize(depth_frame1).get_data())
        # img_scaled = lut_display(depth_image, display_min, display_max)
        # cv2.imshow("test1", img_scaled)

        # img_scaled = cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2RGB)
        img_composite = cv2.addWeighted(color_data, 0.5, colorized_depth, 0.5, 0)
        cv2.imshow("image", img_composite)

        key = cv2.waitKey(4)
        if key == 27:  # esc
            break

    cv2.destroyAllWindows()

    '''color_map = rs.colorizer()
    pipe = rs.pipeline()
    profile = pipe.start()
    data = pipe.wait_for_frames().apply_filter(color_map)
    cv2.imshow("test", data)'''

    '''pipe = rs.pipeline()
    profile = pipe.start()
    try:
        for i in range(0, 100):
            frames = pipe.wait_for_frames()
            for f in frames:
                print(f.profile)
    finally:
        pipe.stop()'''

    '''cap = cv2.VideoCapture("v4l2src device=/dev/video4 ! video/x-raw, format=GRAY8, width=640, heigh=480 ! appsink", cv2.CAP_GSTREAMER)
    ret, frame = cap.read()
    while(ret):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.imshow("test", frame)
        key = cv2.waitKey(4)
        if key == 27:  # esc
            break'''