import time

import cv2
import os
import sys
#os.system("\"c:\\Program Files (x86)\\IntelSWTools\\openvino\\bin\\setupvars.bat\"")
from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np

#vcap = cv2.VideoCapture("rtsp://admin:LABCC0805$@192.168.1.64")#/Streaming/Channels/102")
vcap = cv2.VideoCapture(0)#/Streaming/Channels/102")
#vcap = cv2.VideoCapture("rtspsrc location=rtsp://admin:LABCC0805$@192.168.1.64 latency=0 ! rtph265depay ! h265parse ! omxh265dec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
#vcap.set(cv2.CAP_PROP_FPS, 1)
vcap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
vcap.set(cv2.CAP_PROP_POS_FRAMES, 1)

device = "MYRIAD"
cpu_extension = "extension\cpu_extension.dll"
model_xml = "vino/example2/models_fp16/face-detection-retail-0004.xml"
model_bin = "vino/example2/models_fp16/face-detection-retail-0004.bin" #os.path.splitext(model_xml)[0] + ".bin"
plugin_dir = None
prob_threshold_face = 0.5
cur_request_id = 0
#next_request_id = 1

devices = [device, device]

# Create detectors class instance
#detections = interactive_detection.Detections(devices, models, args.plugin_dir, args.prob_threshold, args.prob_threshold_face, is_async_mode)
#self.face_detectors = detectors.FaceDetection(self.device_fc, self.model_fc, self.cpu_extension, self.plugin_dir, self.prob_threshold_face, self.is_async_mode)
#super().__init__(device, model_xml, cpu_extension, plugin_dir, detection_of)
plugin = None
if device == 'MYRIAD':
    #self.plugin = self._init_plugin(device, plugin_dir)
    plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
    '''if cpu_extension and 'CPU' in device:
        plugin.add_cpu_extension(cpu_extension)'''
'''else:
    #self.plugin = self._init_plugin(device, plugin_dir)
    plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
    if cpu_extension and 'CPU' in device:
        plugin.add_cpu_extension(cpu_extension)'''
# Read IR
#self.net = self._read_ir(model_xml, detection_of)
net = IENetwork(model=model_xml, weights=model_bin)
# Load IR model to the plugin
#self.input_blob, self.out_blob, self.exec_net, self.input_dims, self.output_dims = self._load_ir_to_plugin(device, detection_of)
'''if device == "CPU":
    supported_layers = plugin.get_supported_layers(net)
    not_supported_layers = [
        l for l in net.layers.keys() if l not in supported_layers
    ]
    if len(not_supported_layers) != 0:
        sys.exit(1)'''

assert len(net.inputs.keys()) == 1, "Face Detection network should have only one input"
assert len(net.outputs) == 1, "Face Detection network should have only one output"

input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
exec_net = plugin.load(network=net, num_requests=2)
input_dims = net.inputs[input_blob].shape
output_dims = net.outputs[out_blob].shape
n, c, h, w = input_dims

while(1):
    #print("------- " + str(time.process_time()))
    ret, face_frame = vcap.read()
    #print("vcap.read " + str(time.process_time()))
    if not ret:
        continue

    #self.face_detectors.submit_req(frame, next_frame, is_async_mode)
    exec_net.requests[cur_request_id].wait(-1)
    #print("exec_net.requests[cur_request_id].wait(-1) " + str(time.process_time()))
    in_frame = cv2.resize(face_frame, (w, h))
    #print("cv2.resize " + str(time.process_time()))
    # Change data layout from HWC to CHW
    in_frame = in_frame.transpose((2, 0, 1))
    #print("in_frame.transpose " + str(time.process_time()))
    in_frame = in_frame.reshape((n, c, h, w))
    #print("in_frame.reshape " + str(time.process_time()))
    exec_net.start_async(
        request_id=cur_request_id,
        inputs={input_blob: in_frame})
    #print("exec_net.start_async " + str(time.process_time()))
    #ret = self.face_detectors.wait()
    if not exec_net.requests[cur_request_id].wait(-1) == 0:
        cv2.imshow('VIDEO', face_frame)
        cv2.waitKey(1)
        continue
    #print("if not exec_net.requests[cur_request_id].wait(-1) == 0 " + str(time.process_time()))
    # faces = self.face_detectors.get_results(is_async_mode)
    '''
    The net outputs a blob with shape: [1, 1, 200, 7]
    The description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
    '''
    res = exec_net.requests[cur_request_id].outputs[out_blob]  # res's shape: [1, 1, 200, 7]
    # Get rows whose confidence is larger than prob_threshold.
    # detected faces are also used by age/gender, emotion, landmark, head pose detection.
    faces = res[0][:, np.where(res[0][0][:, 2] > prob_threshold_face)]
    #print("faces = res[0][:, np.where(res[0][0][:, 2] > prob_threshold_face)] " + str(time.process_time()))

    '''if is_async_mode:
        cur_request_id, next_request_id = next_request_id, cur_request_id'''

    bboxes = []
    for i in range(faces.shape[2]):
        face = faces[0][0][i]
        bbox = face[3:7] * np.array([face_frame.shape[1], face_frame.shape[0], face_frame.shape[1], face_frame.shape[0]])
        xmin, ymin, xmax, ymax = bbox.astype("int")
        cv2.rectangle(face_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    #print("cv2.rectangle " + str(time.process_time()))
    cv2.imshow('VIDEO', face_frame)
    #print("cv2.imshow " + str(time.process_time()))
    cv2.waitKey(1)
    #print("cv2.waitKey " + str(time.process_time()))
    #int(round(time.time() * 1000))
