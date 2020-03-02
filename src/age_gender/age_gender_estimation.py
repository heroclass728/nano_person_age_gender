import numpy as np
import cv2
import time
import dlib

from keras.preprocessing import image
from src.age_gender.age_gender_model import age_model, gender_model
from src.settings import MALE_ICON_PATH, FEMALE_ICON_PATH, LOCAL, TRACK_CYCLE, MARGIN, VIDEO_PATH
from src.utils.folder_file_manager import log_print


class FaceDetector:

    def __init__(self):

        self.face_info = {
            "id": [],
            "encoding": [],
            "age": [],
            "gender": [],
            "t_stamp": [],
            "type": [],
            "x": [],
            "y": [],
            "w": [],
            "h": [],
        }
        # self.face_cascade = cv2.CascadeClassifier(FRONT_FACE_PATH)
        self.enable_gender_icons = True
        # you can find male and female icons here: https://github.com/serengil/tensorflow-101/tree/master/dataset

        male_icon = cv2.imread(MALE_ICON_PATH)
        self.male_icon = cv2.resize(male_icon, (40, 40))

        female_icon = cv2.imread(FEMALE_ICON_PATH)
        self.female_icon = cv2.resize(female_icon, (40, 40))
        # -----------------------

        self.age_mdl = age_model()
        self.gender_mdl = gender_model()

        # age model has 101 outputs and its outputs will be multiplied by its index label. sum will be apparent age
        self.output_indexes = np.array([i for i in range(0, 101)])
        self.detector = dlib.get_frontal_face_detector()
        self.face_trackers = {}
        self.current_face_id = 1
        self.face_names = {}
        self.faceAttributes = {}

    def detect_faces(self, img):

        rects = self.detector(img, 1)
        for rect in rects:
            left = rect.left()
            top = rect.top()
            right = rect.right()
            bottom = rect.bottom()
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            x_bar = 0.5 * (left + right)
            y_bar = 0.5 * (top + bottom)

            matched_fid = None

            for fid in self.face_trackers.keys():

                tracked_position = self.face_trackers[fid].get_position()
                t_left = int(tracked_position.left())
                t_top = int(tracked_position.top())
                t_right = int(tracked_position.right())
                t_bottom = int(tracked_position.bottom())

                # calculate the center point
                t_x_bar = 0.5 * (t_left + t_right)
                t_y_bar = 0.5 * (t_top + t_bottom)

                # check if the center point of the face is within the rectangleof a tracker region.
                # Also, the center point of the tracker region must be within the region detected as a face.
                # If both of these conditions hold we have a match

                if ((t_left <= x_bar <= (t_left + t_right)) and (t_top <= y_bar <= (t_top + t_bottom)) and
                        (left <= t_x_bar <= right) and (top <= t_y_bar <= bottom)):
                    matched_fid = fid
                    # If no matched fid, then we have to create a new tracker
            if matched_fid is None:
                print("Creating new tracker " + str(self.current_face_id))
                # Create and store the tracker
                tracker = dlib.correlation_tracker()
                tracker.start_track(img, dlib.rectangle(left - MARGIN, top - MARGIN, right + MARGIN,
                                                        bottom + MARGIN))
                self.face_trackers[self.current_face_id] = tracker
                age, gender = self.detect_age_gender(x=left, y=top, w=right-left, h=bottom-top, img=img)
                # time.sleep(0.1)
                self.faceAttributes[self.current_face_id] = [str(self.current_face_id), age, gender]

                # Increase the currentFaceID counter
                self.current_face_id += 1

        return rects

    def track_faces(self, face_image):

        for fid in self.face_trackers.keys():

            tracked_position = self.face_trackers[fid].get_position()
            t_left = int(tracked_position.left())
            t_top = int(tracked_position.top())
            t_right = int(tracked_position.right())
            t_bottom = int(tracked_position.bottom())

            cv2.rectangle(face_image, (t_left, t_top), (t_right, t_bottom), (0, 0, 255), 2)

            if fid in self.faceAttributes.keys():
                cv2.putText(face_image, 'Id_{}'.format(self.faceAttributes[fid][0]), (int(t_right), int(t_top)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(face_image, 'Age: {}'.format(self.faceAttributes[fid][1]), (int(t_right), int(t_top) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(face_image, 'Gender: {}'.format(self.faceAttributes[fid][2]),
                            (int(t_right), int(t_top) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            else:

                cv2.putText(face_image, "New Person...", (int(t_left), int(t_top)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
        return face_image

    def main(self):

        # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(VIDEO_PATH)
        cnt = 0

        while True:

            ret, img = cap.read()
            # img = self.detect_one_frame(img=img)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            result_image = img.copy()
            cnt += 1
            fids_to_delete = []
            for fid in self.face_trackers.keys():
                tracking_quality = self.face_trackers[fid].update(img)

                # If the tracking quality is good enough, we must delete
                # this tracker
                if tracking_quality < 7:
                    fids_to_delete.append(fid)

            for fid in fids_to_delete:
                print("Removing fid " + str(fid) + " from list of trackers")
                self.face_trackers.pop(fid, None)
                self.faceAttributes.pop(fid, None)

            if cnt % TRACK_CYCLE == 0:
                self.detect_faces(img=img)
            else:
                result_image = self.track_faces(face_image=result_image)

            if LOCAL:
                cv2.imshow("image", result_image)
                time.sleep(0.1)

                if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                    break
        # kill open cv things
        cap.release()
        cv2.destroyAllWindows()

    def detect_age_gender(self, x, y, w, h, img):

        apparent_age = None
        gender = None
        # extract detected face
        detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face

        try:
            # age gender data set has 40% margin around the face. expand detected face.
            margin = 30
            margin_x = int((w * margin) / 100)
            margin_y = int((h * margin) / 100)
            detected_face = img[int(y - margin_y):int(y + h + margin_y),
                                int(x - margin_x):int(x + w + margin_x)]

        except Exception as e:
            log_print(e)
            # print("detected face has no margin")
            # print(e)

        try:
            # vgg-face expects inputs (224, 224, 3)
            detected_face = cv2.resize(detected_face, (224, 224))

            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            # find out age and gender
            age_distributions = self.age_mdl.predict(img_pixels)
            apparent_age = int(np.floor(np.sum(age_distributions * self.output_indexes, axis=1))[0])

            gender_distribution = self.gender_mdl.predict(img_pixels)[0]
            gender_index = np.argmax(gender_distribution)
            if gender_index == 0:
                gender = "F"
            else:
                gender = "M"

        except Exception as e:
            log_print(e)

        return apparent_age, gender


if __name__ == '__main__':
    FaceDetector().main()
