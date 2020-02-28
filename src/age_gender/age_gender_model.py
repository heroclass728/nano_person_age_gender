from src.age_gender.model_loading import load_vgg_face_model
from keras.models import Model, Sequential
from keras.layers import Convolution2D, Flatten, Activation
from src.settings import AGE_MODEL_PATH, GENDER_MODEL_PATH


def age_model():

    model = load_vgg_face_model()

    base_model_output = Sequential()
    base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    age_model_ = Model(inputs=model.input, outputs=base_model_output)

    # you can find the pre-trained weights for age prediction here:
    # https://drive.google.com/file/d/1YCox_4kJ-BYeXq27uUbasu--yz28zUMV/view?usp=sharing
    age_model_.load_weights(AGE_MODEL_PATH)

    return age_model_


def gender_model():

    model = load_vgg_face_model()

    base_model_output = Sequential()
    base_model_output = Convolution2D(2, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    gender_model_ = Model(inputs=model.input, outputs=base_model_output)

    # you can find the pre-trained weights for gender prediction here:
    # https://drive.google.com/file/d/1wUXRVlbsni2FN9-jkS_f4UTUrm1bRLyk/view?usp=sharing
    gender_model_.load_weights(GENDER_MODEL_PATH)

    return gender_model_


if __name__ == '__main__':

    age = age_model()
    gender = gender_model()
