import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
import warnings

warnings.filterwarnings("ignore")

from keras.layers import Dense, Convolution1D, MaxPool1D, Flatten, Dropout
from keras.layers import Input
from keras.models import Model

# from keras.layers.normalization import BatchNormalization
from tensorflow import keras
from keras.layers import BatchNormalization
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

from datetime import datetime


class Solution_ekg_2:
    def __init__(self, shared_definitions, Logger):
        self.Logger = Logger

        self.Logger.info(self, "Creating object of type solution_ekg_2")
        self.shared_definitions = shared_definitions
        self.project_solution_model_filename = (
            shared_definitions.project_solution_ekg_2_model_filename
        )
        self.project_solution_training_script = (
            shared_definitions.project_solution_ekg_2_training_script
        )

    def create_model(self, im_shape):  # im_shape = (X_train.shape[1], 1)
        time_model_begin = datetime.now()

        inputs_cnn = Input(shape=(im_shape), name="inputs_cnn")
        conv1_1 = Convolution1D(64, (6), activation="relu", input_shape=im_shape)(
            inputs_cnn
        )
        conv1_1 = BatchNormalization()(conv1_1)
        pool1 = MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1_1)
        conv2_1 = Convolution1D(64, (3), activation="relu", input_shape=im_shape)(pool1)

        conv2_1 = BatchNormalization()(conv2_1)
        pool2 = MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv2_1)
        conv3_1 = Convolution1D(64, (3), activation="relu", input_shape=im_shape)(pool2)

        conv3_1 = BatchNormalization()(conv3_1)
        pool3 = MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv3_1)
        flatten = Flatten()(pool3)

        dense_end1 = Dense(64, activation="relu")(flatten)
        dense_end2 = Dense(32, activation="relu")(dense_end1)
        main_output = Dense(5, activation="softmax", name="main_output")(dense_end2)

        self.model = Model(inputs=inputs_cnn, outputs=main_output)
        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        time_model = (datetime.now() - time_model_begin).seconds
        info_message = f"Took {time_model} seconds to create model."
        self.Logger.info(self, info_message)

    def train(self, X_train, Y_train, X_val, Y_val, epochs=40):
        f1_time = datetime.now()

        info_message = "Begining training"
        self.Logger.info(self, info_message)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8),
            ModelCheckpoint(
                filepath=self.project_solution_model_filename,
                monitor="val_loss",
                save_best_only=True,
            ),
        ]

        history = self.model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            callbacks=callbacks,
            batch_size=32,
            validation_data=(X_val, Y_val),
        )

        f2_time = datetime.now()
        difference = f2_time - f1_time
        seconds_in_day = 24 * 60 * 60
        divmod(difference.days * seconds_in_day + difference.seconds, 60)
        info_message = "Training - it took {time} for {number} epochs".format(
            time=difference, number=epochs
        )
        self.Logger.info(self, info_message)

        return history

    def save_model(self, filename="", path=""):
        MODEL_SAVE_PATH = ""
        if filename != "" and path != "":
            MODEL_SAVE_PATH += filename + path
        elif filename != "":
            MODEL_SAVE_PATH += filename
        else:
            MODEL_SAVE_PATH = self.project_solution_model_filename

        MODEL_SAVE_PATH = (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + MODEL_SAVE_PATH
        )
        info_message = "Saving model at " + MODEL_SAVE_PATH
        self.Logger.info(self, info_message)
        self.model.save(MODEL_SAVE_PATH)

    def load_model(self, filename="", path=""):
        if filename != "" and path != "":
            pass
        elif filename != "":
            pass
        else:
            self.model = keras.models.load_model(
                self.shared_definitions.project_solution_ekg_2_model_filename_last_good_one
            )

    def test(self):
        pass
