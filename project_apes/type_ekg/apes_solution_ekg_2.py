# This file:
# Contains the solution_ekg_2. Builds a CNN using the 'building blocks' Tensorflow & Keras offer.
#
# Main methods to be called from apes_application:
# create_model(), load_model()
#       -- purpose: self.model exists
# save_model()
#       -- purpose: self.model save
# train(epochs), test()
#       -- purpose: train and use self.model
# adapt_dataset(self, application_instance_metadata, list_of_dataFrames, list_of_dataFramesUtilityLabels)
#       -- purpose: self.X_train, self.y_train, self.X_test, self.y_test etc exist

# TODO: how many output neurons bro

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
from keras.layers import Dense, Convolution1D, MaxPool1D, Flatten, Dropout
from keras.layers import Input
from keras.models import Model

# from keras.layers.normalization import BatchNormalization
from tensorflow import keras
from keras.layers import BatchNormalization
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

from datetime import datetime

import sys

sys.path.insert(1, "../helpers_aiders_and_conveniencers")
from helpers_aiders_and_conveniencers.misc_functions import get_last_model

# TODO: add gaussian noise?
# TODO: resample?


class Solution_ekg_2:
    def __init__(self, app_instance_metadata, Logger):
        self.Logger = Logger

        self.Logger.info(self, "Creating object of type solution_ekg_2")
        self.app_instance_metadata = app_instance_metadata
        self.project_solution_model_filename = (
            app_instance_metadata.shared_definitions.project_solution_ekg_2_model_filename
        )

    ## -------------- General methods --------------
    ## Model methods
    def create_model(self):  # im_shape = (X_train.shape[1], 1)
        info_message = "create_model() -- begin"
        self.Logger.info(self, info_message)

        time_model_begin = datetime.now()

        im_shape = (self.X_train.shape[1], 1)

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
        info_message = f"Took {time_model} seconds to create model"
        self.Logger.info(self, info_message)

        info_message = "create_model() -- end"
        self.Logger.info(self, info_message)

        return 0, f"{self} -- create_model() completed successfully"

    def save_model(self, filename="", path=""):
        info_message = "save_model() -- begin"
        self.Logger.info(self, info_message)

        MODEL_SAVE_PATH = ""
        if filename != "" and path != "":
            MODEL_SAVE_PATH += filename + path
        elif filename != "":
            MODEL_SAVE_PATH += filename
        else:
            MODEL_SAVE_PATH = (
                "s_ekg2_d_"
                + self.app_instance_metadata.dataset_metadata.dataset_name_stub
            )

        MODEL_SAVE_PATH = (
            MODEL_SAVE_PATH + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        info_message = "Saving model as " + MODEL_SAVE_PATH
        self.Logger.info(self, info_message)
        self.model.save(MODEL_SAVE_PATH)

        info_message = "save_model() -- end"
        self.Logger.info(self, info_message)

        return 0, f"{self} -- save_model() completed successfully"

    def load_model(self, filename="", path=""):
        info_message = "load_model() -- begin"
        self.Logger.info(self, info_message)

        if filename != "" and path != "":
            pass
        elif filename != "":
            pass
        else:
            try:
                info_message = "Trying to load the last good model saved"
                self.Logger.info(self, info_message)
                self.model = keras.models.load_model(
                    self.app_instance_metadata.shared_definitions.project_solution_ekg_2_model_filename_last_good_one,
                    map_location=lambda storage, loc: storage.cuda(1),
                )
            except:
                info_message = "Failed to load the last saved good model. Could be an error, or there is none saved. Loading last available model"
                self.Logger.info(self, info_message)
                return_code, return_message, model_path = get_last_model(
                    self.Logger, "ekg2", self.app_instance_metadata
                )
                if return_code != 0:
                    self.Logger.info(self, return_message)
                    return
                else:
                    self.model = keras.models.load_model(
                        model_path, map_location=lambda storage, loc: storage.cuda(0)
                    )

        info_message = "load_model() -- end"
        self.Logger.info(self, info_message)

        return 0, f"{self} -- load_model() completed successfully"

    ## Functionality methods
    def train(self, epochs=40):
        info_message = "train() -- begin"
        self.Logger.info(self, info_message)

        info_message = "##########################################################"
        self.Logger.info(self, info_message)
        info_message = (
            f"Begining training for solution_ekg_2. Number of epochs: {epochs}"
        )
        self.Logger.info(self, info_message)
        info_message = "##########################################################"
        self.Logger.info(self, info_message)

        f1_time = datetime.now()
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8),
            ModelCheckpoint(
                filepath=self.project_solution_model_filename,
                monitor="val_loss",
                save_best_only=True,
            ),
        ]

        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            callbacks=callbacks,
            batch_size=32,
            validation_data=(self.X_test, self.y_test),
        )

        f2_time = datetime.now()
        difference = f2_time - f1_time
        seconds_in_day = 24 * 60 * 60
        divmod(difference.days * seconds_in_day + difference.seconds, 60)
        info_message = f"Training - it took {difference} for {epochs} epochs"
        self.Logger.info(self, info_message)

        info_message = "train() -- end"
        self.Logger.info(self, info_message)

        return 0, f"{self} -- train() completed successfully"
        return history

    def test(self):
        info_message = "Begining testing"
        self.Logger.info(self, info_message)
        info_message = f"Testing - it took "
        self.Logger.info(self, info_message)

        return 0, f"{self} -- test() completed successfully"

    ## Dataset methods
    def adapt_dataset(
        self,
        application_instance_metadata,
        list_of_dataFrames,
        list_of_dataFramesUtilityLabels,
    ):
        info_message = "adapt_dataset() -- begin"
        self.Logger.info(self, info_message)

        try:
            train_df = list_of_dataFrames[
                list_of_dataFramesUtilityLabels.index("train")
            ]
            cols = train_df.shape[1]
            if application_instance_metadata.dataset_metadata.is_labeled == True:
                cols = cols - 1
                target_train = [int(x) for x in train_df.iloc[:, cols].values]
                if min(target_train) > 0:
                    if min(target_train) == 1:
                        target_train = [x - 1 for x in target_train]
                    else:
                        minimum = min(target_train)
                        target_train = [x - minimum for x in target_train]
                self.y_train = to_categorical(target_train)
            X_train = train_df.iloc[:, : cols - 1].values
            self.X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
            info_message = f"Created X_train (shape {self.X_train.shape}) and y_train (shape {self.y_train.shape})"
            self.Logger.info(self, info_message)
        except:
            info_message = "No train dataFrame"
            self.Logger.info(self, info_message)

        try:
            test_df = list_of_dataFrames[list_of_dataFramesUtilityLabels.index("test")]
            cols = test_df.shape[1]
            if application_instance_metadata.dataset_metadata.is_labeled == True:
                cols = cols - 1
                target_test = [int(x) for x in test_df.iloc[:, cols].values]
                if min(target_test) > 0:
                    if min(target_test) == 1:
                        target_test = [x - 1 for x in target_test]
                    else:
                        minimum = min(target_test)
                        target_test = [x - minimum for x in target_test]
                self.y_test = to_categorical(target_test)
            X_test = test_df.iloc[:, : cols - 1].values
            self.X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)
            info_message = f"Created X_test (shape {self.X_test.shape}) and y_test (shape {self.y_test.shape})"
            self.Logger.info(self, info_message)
        except:
            info_message = "No test dataFrame"
            self.Logger.info(self, info_message)

        try:
            val_df = list_of_dataFrames[list_of_dataFramesUtilityLabels.index("val")]
            cols = val_df.shape[1]
            if application_instance_metadata.dataset_metadata.is_labeled == True:
                cols = cols - 1
                target_val = [int(x) for x in val_df.iloc[:, cols].values]
                self.y_val = to_categorical(target_val)
            X_val = val_df.iloc[:, : cols - 1].values
            self.X_val = X_val.reshape(len(X_val), X_val.shape[1], 1)
            info_message = f"Created X_val (shape {self.X_val.shape}) and y_val (shape {self.y_val.shape})"
            self.Logger.info(self, info_message)
        except:
            info_message = "No val dataFrame"
            self.Logger.info(self, info_message)

        try:
            run_df = list_of_dataFrames[list_of_dataFramesUtilityLabels.index("run")]
            cols = run_df.shape[1]
            if application_instance_metadata.dataset_metadata.is_labeled == True:
                cols = cols - 1
                target_run = [int(x) for x in run_df.iloc[:, cols].values]
                self.y_run = to_categorical(target_run)
            X_run = run_df.iloc[:, : cols - 1].values
            self.X_run = X_run.reshape(len(X_run), X_run.shape[1], 1)
            info_message = f"Created X_run (shape {self.X_run.shape}) and y_run (shape {self.y_run.shape})"
            self.Logger.info(self, info_message)
        except:
            info_message = "No run dataFrame"
            self.Logger.info(self, info_message)

        info_message = "adapt_dataset() -- end"
        self.Logger.info(self, info_message)

        return 0, f"{self} -- adapt_dataset() completed successfully"

    ## -------------- Particular methods --------------
