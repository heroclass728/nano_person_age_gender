import numpy as np
import cv2

from keras.preprocessing import image
from src.age_gender.age_gender_model import age_model, gender_model
from src.settings import MALE_ICON_PATH, FEMALE_ICON_PATH, FRONT_FACE_PATH, LOCAL
from src.utils.folder_file_manager import log_print


class AgeGenderDetector:

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
        self.face_cascade = cv2.CascadeClassifier(FRONT_FACE_PATH)
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

    def detect_age_gender(self):

        cap = cv2.VideoCapture(0)

        while True:

            ret, img = cap.read()
            img = self.detect_one_frame(img=img)

            # self.face_info = self.customize_on_time()
            if LOCAL:
                cv2.imshow("image", img)

                if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                    break
        # kill open cv things
        cap.release()
        cv2.destroyAllWindows()

    def detect_one_frame(self, img):

        faces = self.face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in faces:
            if w > 50:  # ignore small faces

                # mention detected face
                """overlay = img.copy(); output = img.copy(); opacity = 0.6
                cv2.rectangle(img,(x,y),(x+w,y+h),(128,128,128),cv2.FILLED) #draw rectangle to main image
                cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)"""
                cv2.rectangle(img, (x, y), (x + w, y + h), (128, 128, 128), 1)  # draw rectangle to main image

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
                    # face_boxes.append(face_box)
                    # ages.append(apparent_age)
                    if LOCAL:

                        enable_gender_icons = True
                        if gender_index == 0:
                            gender = "F"
                        else:
                            gender = "M"

                        # background for age gender declaration
                        info_box_color = (46, 200, 255)
                        # triangle_cnt = np.array([(x+int(w/2), y+10), (x+int(w/2)-25, y-20),
                        # (x+int(w/2)+25, y-20)])
                        triangle_cnt = np.array(
                            [(x + int(w / 2), y), (x + int(w / 2) - 20, y - 20), (x + int(w / 2) + 20, y - 20)])
                        cv2.drawContours(img, [triangle_cnt], 0, info_box_color, -1)
                        cv2.rectangle(img, (x + int(w / 2) - 50, y - 20), (x + int(w / 2) + 50, y - 90),
                                      info_box_color, cv2.FILLED)

                        # labels for age and gender
                        cv2.putText(img, str(apparent_age), (x + int(w / 2), y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 111, 255), 2)

                        if enable_gender_icons:
                            if gender == 'M':
                                gender_icon = self.male_icon
                            else:
                                gender_icon = self.female_icon

                            img[y - 75:y - 75 + self.male_icon.shape[0],
                                x + int(w / 2) - 45:x + int(w / 2) - 45 + self.male_icon.shape[1]] = gender_icon
                        else:
                            cv2.putText(img, gender, (x + int(w / 2) - 42, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 111, 255), 2)

                except Exception as e:
                    log_print(e)
                    # print("exception", str(e))

        return img


if __name__ == '__main__':
    AgeGenderDetector().detect_age_gender()
