import os

from utils.folder_file_manager import make_directory_if_not_exists


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PERSON_MODEL_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'utils', 'model', 'person'))

CAFFE_MODEL = os.path.join(PERSON_MODEL_DIR, 'MobileNetSSD_deploy.caffemodel')
CAFFE_PROTEXT = os.path.join(PERSON_MODEL_DIR, 'MobileNetSSD_deploy.prototxt')

DETECTION_CONFIDENT = 0.4
SKIP_FRAMES = 30
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
