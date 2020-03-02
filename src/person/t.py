import base64
import json
import cv2
import os
import sys
import threading
import time
import datetime
import numpy as np
import redis
from paho.mqtt.client import Client


_cur_dir = os.path.dirname(os.path.realpath(__file__))
if _cur_dir not in sys.path:
    sys.path.append(_cur_dir)

from utils.object_track import ObjectTrack
from utils.nodered import get_flow
from utils.cv import get_contour_list
from settings import NODERED_FLOW_TIMEOUT, DOWNLOAD_MODEL_TIMEOUT, REDIS_PREFIX, MQTT_PREFIX, SAVE_IMAGE, SAVE_PATH
from utils.common import logger
from utils.models import download_model


client = Client()


class VisoODService(threading.Thread):

    config = {}

    def __init__(self):
        super().__init__()
        client.on_connect = on_mqtt_connected
        client.on_message = on_mqtt_message
        client.connect('127.0.0.1')
        client.loop_start()
        self.sources = []
        self._last_saved_time = []

    def _parse_nodered_flow(self):
        flow = get_flow()
        if flow:
            for f in flow['flows']:
                if f['type'] == 'object-detect':
                    self.config = f
                    logger.info(f'Parsed flow - {f}')
                    self._get_sources(flow)
                    return True

    def _get_sources(self, flow):

        def __find_parent_cf_node(node):
            if node['type'] == 'camera-feed':
                return node
            for _node in flow['flows']:
                if _node.get('wires') and node['id'] in _node['wires'][0] and _node['type'] != 'object-detect':
                    return __find_parent_cf_node(_node)

        for f in flow['flows']:
            if f.get('wires') and self.config['id'] in f['wires'][0]:
                cf_node = __find_parent_cf_node(f)
                logger.info(f'Found a parent & source CF node - {f["id"]} & {cf_node["id"]}')
                f['roi_list'] = get_contour_list(rois=cf_node.get('rois', []),
                                                 width=cf_node['frame_width'], height=cf_node['frame_height'])
                self.sources.append(f)
                self._last_saved_time.append(0)

    def run(self):
        # Download the flow from NodeRed container and parse.
        s_time = time.time()
        while True:
            if self._parse_nodered_flow():
                break
            else:
                if time.time() - s_time > NODERED_FLOW_TIMEOUT * 60:
                    logger.critical('Failed to download flow, exiting...')
                    return
                else:
                    time.sleep(1)

        s_time = time.time()
        model_name = self.config.get('model_name')
        detect_mode = str(self.config.get('detect_mode', '')).lower()
        model_url = self.config.get('custom_model_url')
        while True:
            model_dir = download_model(model_name=model_name, device=detect_mode,
                                       model_url=model_url if self.config.get('public_model', True) is False else None)
            if model_dir:
                break
            else:
                if time.time() - s_time > DOWNLOAD_MODEL_TIMEOUT * 60:
                    logger.critical('Failed to download model, existing...')
                    return
                else:
                    time.sleep(.1)

        if detect_mode == 'gpu':
            if model_name == 'yolov3':
                from utils.object_detect_gpu_yolov3 import VisoGPUODYoloV3
                detector = VisoGPUODYoloV3(model_dir=model_dir)
            else:
                from utils.object_detect_gpu import VisoGPUOD
                detector = VisoGPUOD(model_dir=model_dir)
        else:
            if model_name == 'yolov3':
                from utils.openvino_detect_yolov3 import OpenVINODetectYOLOV3
                detector = OpenVINODetectYOLOV3(
                    model_dir=model_dir, device='MYRIAD' if detect_mode == 'ncs' else 'CPU')
            else:
                if model_url:
                    from utils.object_detect_gpu import VisoGPUOD
                    detector = VisoGPUOD(model_dir=model_dir)
                else:
                    from utils.openvino_detect import OpenVinoObjectDetect
                    detector = OpenVinoObjectDetect(
                        model_dir=model_dir, device='MYRIAD' if detect_mode == 'ncs' else 'CPU')

        tracking_mode = str(self.config.get('tracking_algorithm')).upper()
        tracking_quality = float(self.config.get('tracking_quality', 5))
        tracking_cycle = int(self.config.get('tracking_cycle', 2))

        trackers = []
        cnts = []
        for vid_src_id in range(len(self.sources)):
            trackers.append(ObjectTrack(trk_type=tracking_mode, good_track_quality=tracking_quality))
            cnts.append(0)

        logger.info("Starting detection loop...")
        r = redis.StrictRedis()
        while True:
            for vid_src_id, src in enumerate(self.sources):
                str_frame = r.get(f"{REDIS_PREFIX}_{src.get('id')}")
                if str_frame:
                    str_frame = base64.b64decode(str_frame)
                    frame = cv2.imdecode(np.fromstring(str_frame, dtype=np.uint8), -1)
                    h, w = frame.shape[:2]
                    if cnts[vid_src_id] % tracking_cycle == 0:
                        result = detector.detect_frame(frame)
                        filtered_objects = [r for r in result if r['label'] in self.config.get('labels', [])]
                        # FIXME: Remove this!
                        if filtered_objects:
                            logger.debug(filtered_objects)
                        cnts[vid_src_id] = 0
                        trackers[vid_src_id].upgrade_trackers(dets=filtered_objects, trk_img=frame)
                    else:
                        trackers[vid_src_id].keep_trackers(trk_img=frame)
                    cnts[vid_src_id] += 1

                    result = trackers[vid_src_id].to_list()

                    roi_result = [
                        x for x in result if
                        not src.get('roi_list', []) or  # ROI is not defined?
                        any([
                            cv2.pointPolygonTest(
                                np.array([cnt], dtype=np.int32),
                                ((x['rect'][0] + x['rect'][2] // 2) * w,
                                 (x['rect'][1] + x['rect'][3] // 2) * h),  # Center point
                                False) >= 0
                            for cnt in src.get('roi_list', [])
                        ])
                    ]
                    if roi_result:
                        logger.info(f'Detected Object from {src.get("id")}({src.get("name")}) - {roi_result}')
                        client.publish(topic=f"{MQTT_PREFIX}_{self.config['id']}",
                                       payload=json.dumps({
                                           "camera_id": src.get("id"),
                                           "result": roi_result
                                       }))
                        if SAVE_IMAGE and time.time() - self._last_saved_time[vid_src_id] > 60:
                            f_name = os.path.join(
                                SAVE_PATH, f"{src.get('id')}_{datetime.datetime.now().isoformat()}.jpg")
                            logger.debug(f"Saving to a file - {f_name}")
                            for r in roi_result:
                                _x, _y, _w, _h = (np.array(r['rect']) * np.array([w, h, w, h])).astype(np.int).tolist()
                                cv2.rectangle(frame, (_x, _y), (_x + _w, _y + _h), (0, 255, 0), 2)
                                cv2.putText(frame, f'{r["label"]} {round(r["confidence"] * 100, 1)} %', (_x, _y - 7),
                                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
                            cv2.imwrite(f_name, frame)
                            self._last_saved_time[vid_src_id] = time.time()


def on_mqtt_connected(*args):
    logger.info(f'Connected to the MQTT broker - {args}')


def on_mqtt_message(*args):
    # topic = args[2].topic
    # msg = args[2].payload.decode('utf-8')
    logger.info(f'Received a message - {args}')


if __name__ == '__main__':

    logger.info('========== Staring Viso OD Service ==========')

    od = VisoODService()
    od.start()
