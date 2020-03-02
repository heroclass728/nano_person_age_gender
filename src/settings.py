import os

from src.utils.folder_file_manager import make_directory_if_not_exists


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PERSON_MODEL_DIR = os.path.join(CUR_DIR, 'person')
# os.makedirs(PERSON_MODEL_DIR, exist_ok=True)

CAFFE_MODEL = os.path.join(PERSON_MODEL_DIR, 'MobileNetSSD_deploy.caffemodel')
CAFFE_PROTEXT = os.path.join(PERSON_MODEL_DIR, 'MobileNetSSD_deploy.prototxt')
MODEL_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'age_gender'))
AGE_MODEL_PATH = os.path.join(MODEL_DIR, 'age_model_weights.h5')
GENDER_MODEL_PATH = os.path.join(MODEL_DIR, 'gender_model_weights.h5')

DETECTION_CONFIDENT = 0.4
SKIP_FRAMES = 30
TRACK_CYCLE = 20
MARGIN = 10
VIDEO_PATH = ""
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

ICON_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'utils', 'icons'))
MALE_ICON_PATH = os.path.join(ICON_DIR, 'male.jpg')
FEMALE_ICON_PATH = os.path.join(ICON_DIR, 'female.jpg')
FRONT_FACE_PATH = os.path.join(CUR_DIR, 'age_gender', "haarcascade_frontalface_default.xml")
LOCAL = True
