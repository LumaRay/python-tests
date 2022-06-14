import os
import shutil
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import json
# import time
from urllib.parse import unquote
# using jel https://github.com/LumaRay/jel

# IMAGES_INPUT_PATH = "C:/Work/InfraredCamera/ThermalView/tests/train_models/parsed_data/v2_vk_sources2/objects/mask"
# IMAGES_OUTPUT_PATH = "C:/Work/InfraredCamera/ThermalView/tests/train_models/parsed_data/v2_vk_sources2/objects/_sorted"

IMAGES_INPUT_PATH = "C:/Work/InfraredCamera/ThermalView/tests/train_models/parsed_data/v2_exhib_source_2021_04_14/objects/back"
IMAGES_OUTPUT_PATH = "C:/Work/InfraredCamera/ThermalView/tests/train_models/parsed_data/v2_exhib_source_2021_04_14/objects/_sorted"

SORT_FOLDERS = ['_del', 'back', 'mask', 'maskchin', 'masknone', 'masknose']

for sort_folder in SORT_FOLDERS:
    if not os.path.exists(IMAGES_OUTPUT_PATH + "/" + sort_folder):
        os.makedirs(IMAGES_OUTPUT_PATH + "/" + sort_folder)

# hostName = "localhost"
hostName = "0.0.0.0"
# hostName = "192.168.1.66"
# hostName = "192.168.1.42"
serverPort = 8080

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        get_params = unquote(self.path).split("/")[1:]
        req_list = get_params[-1].split("?")
        req = req_list[0]
        sort_folder = None
        sort_img = None
        undo = False
        # unlocked = False
        if len(req_list) > 1:
            req_param_list = req_list[1].split("&")
            for req_param in req_param_list:
                req_param_split = req_param.split("=")
                req_param_name = req_param_split[0]
                if req_param_name == "undo":
                    undo = True
                # if req_param_name == "unlock":
                #     unlocked = True
                if len(req_param_split) > 1:
                    req_param_value = req_param_split[1]
                    if req_param_name == "sort":
                        sort_folder = req_param_value
                    if req_param_name == "img":
                        sort_img = req_param_value
        # if not unlocked:
        #     self.wfile.write(b"err")
        req_split = req.split(".")
        req_split_file_name = req
        req_split_file_ext = None
        if len(req_split) > 1:
            req_split_file_name = ".".join(req_split[:-1])
            req_split_file_ext = req_split[-1]
        if req == "jel.min.js":
            self.send_response(200)
            self.send_header("Content-type", "text/javascript")
            self.end_headers()
            f = open("jel.min.js", "rb")
            for each_line in f:
                self.wfile.write(each_line)
            return
        elif req == "favicon.ico":
            self.send_response(404)
            self.end_headers()
            return
        elif req_split_file_ext is not None and req_split_file_ext == "jpeg" or req_split_file_ext == "jpg" or req_split_file_ext == "JPG":
            try:
                with open(IMAGES_INPUT_PATH + "/" + req, "rb") as fout:
                    self.send_response(200)
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(fout.read())
            except:
                self.send_response(404)
                self.end_headers()
            return
        else:
            last_moved_image = ""
            last_moved_folder = ""
            if sort_folder is not None and sort_img is not None and sort_img != "":
                try:
                    if undo:
                        if os.path.isfile(IMAGES_INPUT_PATH + "/" + sort_img):
                            os.remove(IMAGES_OUTPUT_PATH + "/" + sort_folder + "/" + sort_img)
                        else:
                            shutil.move(IMAGES_OUTPUT_PATH + "/" + sort_folder + "/" + sort_img, IMAGES_INPUT_PATH)
                    else:
                        if os.path.isfile(IMAGES_OUTPUT_PATH + "/" + sort_folder + "/" + sort_img):
                            os.remove(IMAGES_INPUT_PATH + "/" + sort_img)
                        else:
                            shutil.move(IMAGES_INPUT_PATH + "/" + sort_img, IMAGES_OUTPUT_PATH + "/" + sort_folder)
                except:
                    pass
                last_moved_image = sort_img
                last_moved_folder = sort_folder

            img_filename = ""
            files = os.listdir(IMAGES_INPUT_PATH)
            for fidx, file in enumerate(files):
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".JPG"):
                    img_filename = file
                    break
            doc_struct = [
                {"style": """
                    html, body, body > div {height: 100%}
                    input.sort_but {font-size: 8vw; margin: 3pt; padding: 3pt}
                    form {display: inline}
                """},
                {"div": [
                    {"div": {"style": {"height": "100%"}, "children": [
                        {"p": str(len(files))},
                        {"img": {"style": {"width": "100%", "height": "100%", "object-fit": "contain"}, "src": img_filename}}
                    ]}},
                    {"div": {"style": {"position": "fixed", "bottom": 0}, "children": [
                            {"a": {"href": "?img=" + img_filename + "&sort=" + but, "chi": [{"input": {"type": "submit", "class": "sort_but", "value": but}}]}} for but in SORT_FOLDERS
                        ] + ([
                            {"a": {"href": "?img=" + last_moved_image + "&sort=" + last_moved_folder + "&undo", "chi": [{"input": {"type": "submit", "class": "sort_but", "value": "undo"}}]}}
                        ] if last_moved_image != "" else [])
                    }}
                ]}
            ]

            content_bytes = bytes('<!DOCTYPE html><html><head><script src="/jel.min.js"></script></head><body><script type="text/javascript">document.addEventListener("DOMContentLoaded", function(event) {jel(' + json.dumps(doc_struct) + ')});</script></body></html>', 'utf-8')
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Content-length", str(len(content_bytes)))
            self.end_headers()
            self.wfile.write(content_bytes)

if __name__ == "__main__":
    # webServer = HTTPServer((hostName, serverPort), MyServer)
    # https://stackoverflow.com/questions/43146298/http-request-from-chrome-hangs-python-webserver
    webServer = ThreadingHTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")