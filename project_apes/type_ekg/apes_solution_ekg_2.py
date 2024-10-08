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

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import itertools

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
import json

sys.path.insert(1, "../helpers_aiders_and_conveniencers")
from helpers_aiders_and_conveniencers.misc_functions import (
    get_last_model,
    model_filename_fits_expected_name,
    get_full_path_of_given_model,
    get_plot_save_filename,
    write_to_solutions_runs_json_file,
    get_accuracies_from_confusion_matrix,
    assess_whether_to_save_model_as_best,
)
from helpers_aiders_and_conveniencers.solution_serializer import Solution_Serializer

# TODO: add gaussian noise?
# TODO: resample?


class Solution_ekg_2:
    def __init__(self, app_instance_metadata, Logger):
        self.Logger = Logger
        self.Logger.info(self, "Creating object of type solution_ekg_2")

        self.solution_serializer = Solution_Serializer()
        self.solution_serializer._app_instance_ID = (
            app_instance_metadata.app_instance_ID
        )
        self.solution_serializer._time_object_creation = datetime.now().strftime(
            "%Y-%m-%d_%H:%M:%S"
        )

        self.plots_filenames = []
        # self.datasets_names = []
        self.solution_serializer.solution_name = "ekg2"
        self.solution_serializer.dataset_name = (
            app_instance_metadata.dataset_metadata.dataset_name_stub
        )

        self.app_instance_metadata = app_instance_metadata
        self.project_solution_model_filename = (
            app_instance_metadata.shared_definitions.project_solution_ekg2_model_filename
        )
        self.last_good_suitable_model = ""
        match self.solution_serializer.dataset_name:
            case "ekg1":
                self.last_good_suitable_model = (
                    app_instance_metadata.shared_definitions.project_solution_ekg2_d_ekg1_best_model_filename
                )
            case "ekg2":
                self.last_good_suitable_model = (
                    app_instance_metadata.shared_definitions.project_solution_ekg2_d_ekg2_best_model_filename
                )

        self.ACCURACY_THRESHOLD_PER_CLASS = 70
        self.MEAN_ACCURACY_THRESHOLD = 70

    ## -------------- To JSON --------------
    def toJSON(self):  # this is not used
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def save_run(self):
        info_message = "save_run() -- begin"
        self.Logger.info(self, info_message)

        self.solution_serializer.plots_filenames = self.plots_filenames

        write_to_solutions_runs_json_file(
            self.Logger, "ekg2", self.solution_serializer, self.app_instance_metadata
        )

        info_message = "save_run() -- assess_whether_to_save_model_as_best"
        self.Logger.info(self, info_message)

        try:
            if self.solution_serializer._used_train_function == True:
                assess_whether_to_save_model_as_best(
                    self.Logger,
                    self.app_instance_metadata,
                    "ekg2",
                    self.MODEL_SAVE_PATH.split("/")[-1],
                    self.accuracy_per_class,
                    self.mean_total_accuracy,
                )
        except:
            pass

        info_message = "save_run() -- end"
        self.Logger.info(self, info_message)

        return 0, f"{self} -- save_run() completed successfully"

    ## -------------- General methods --------------
    ## Model methods
    def create_model(self):  # im_shape = (X_train.shape[1], 1)
        info_message = "create_model() -- begin"
        self.Logger.info(self, info_message)

        no_of_classes = self.app_instance_metadata.dataset_metadata.number_of_classes

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
        main_output = Dense(no_of_classes, activation="softmax", name="main_output")(
            dense_end2
        )

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

        self.MODEL_SAVE_PATH = ""
        if filename != "" and path != "":
            self.MODEL_SAVE_PATH += filename + path
        elif filename != "":
            self.MODEL_SAVE_PATH += filename
        else:
            self.MODEL_SAVE_PATH = (
                "s_ekg2_d_"
                + self.app_instance_metadata.dataset_metadata.dataset_name_stub
            )

        self.MODEL_SAVE_PATH = (
            self.MODEL_SAVE_PATH
            + "_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".h5"
        )
        info_message = "Saving model as " + self.MODEL_SAVE_PATH
        self.Logger.info(self, info_message)
        self.model.save(self.MODEL_SAVE_PATH)
        self.solution_serializer.model_filename = self.MODEL_SAVE_PATH.split("/")[-1]

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
            if model_filename_fits_expected_name(
                "ekg2",
                self.app_instance_metadata,
                self.last_good_suitable_model,
            ):
                info_message = "Loading the last good model saved"
                self.Logger.info(self, info_message)
                model_absolute_path = get_full_path_of_given_model(
                    self.last_good_suitable_model,
                    self.app_instance_metadata.shared_definitions.project_model_root_path,
                )
                self.model = keras.models.load_model(model_absolute_path)
                self.solution_serializer.model_filename = model_absolute_path.split(
                    "/"
                )[-1]
            else:
                info_message = "Loading last available model"
                self.Logger.info(self, info_message)
                return_code, return_message, model_path = get_last_model(
                    self.Logger, "ekg2", self.app_instance_metadata
                )
                if return_code != 0:
                    self.Logger.info(self, return_message)
                    return return_code, return_message
                else:
                    self.model = keras.models.load_model(model_path)
                    self.solution_serializer.model_filename = model_path.split("/")[-1]

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
            # ModelCheckpoint(
            #     filepath=self.project_solution_model_filename,
            #     monitor="val_loss",
            #     save_best_only=True,
            # ),
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

        ## Save run serialization data -- begin
        self.solution_serializer.time_train_start = f1_time.strftime(
            "%Y-%m-%d_%H:%M:%S"
        )
        self.solution_serializer._used_train_function = True
        self.solution_serializer.time_train_end = f2_time.strftime("%Y-%m-%d_%H:%M:%S")
        self.solution_serializer.time_train_total = difference.seconds
        self.solution_serializer.train_epochs = epochs
        ## Save run serialization data -- end

        info_message = f"Training - it took {difference} for {epochs} epochs"
        self.Logger.info(self, info_message)

        info_message = "train() -- end"
        self.Logger.info(self, info_message)

        self.solution_serializer.train_loss = history.history["loss"][-1]
        self.solution_serializer.train_accuracy = history.history["accuracy"][-1]
        self.solution_serializer.train_val_loss = history.history["val_loss"][-1]
        self.solution_serializer.train_val_accuracy = history.history["val_accuracy"][
            -1
        ]

        return 0, f"{self} -- train() completed successfully"

    def test(self):
        info_message = "Begining testing"
        self.Logger.info(self, info_message)

        f1_time = datetime.now()

        info_message = "##########################################################"
        self.Logger.info(self, info_message)
        info_message = f"Begining testing for solution_ekg_2"
        self.Logger.info(self, info_message)
        info_message = "##########################################################"
        self.Logger.info(self, info_message)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8),
            # ModelCheckpoint(
            #     filepath=self.project_solution_model_filename,
            #     monitor="val_loss",
            #     save_best_only=True,
            # ),
        ]

        predictions = self.model.predict(
            self.X_test, batch_size=32, callbacks=callbacks, max_queue_size=10
        )

        self.print_results(predictions)

        f2_time = datetime.now()
        difference = f2_time - f1_time
        seconds_in_day = 24 * 60 * 60
        divmod(difference.days * seconds_in_day + difference.seconds, 60)

        sklearn_confusion_matrix = confusion_matrix(
            self.y_test.argmax(axis=1), predictions.argmax(axis=1)
        )

        print(sklearn_confusion_matrix)

        (
            self.accuracy_per_class,
            self.mean_total_accuracy,
        ) = get_accuracies_from_confusion_matrix(
            self.Logger,
            sklearn_confusion_matrix,
            self.ACCURACY_THRESHOLD_PER_CLASS,
            self.MEAN_ACCURACY_THRESHOLD,
        )

        self.plot_confusion_matrix(
            sklearn_confusion_matrix,
            self.app_instance_metadata.dataset_metadata.class_names,
        )

        ## Save run serialization data -- begin
        self.solution_serializer.time_test_start = f1_time.strftime("%Y-%m-%d_%H:%M:%S")
        self.solution_serializer._used_test_function = True
        self.solution_serializer.time_test_end = f2_time.strftime("%Y-%m-%d_%H:%M:%S")
        self.solution_serializer.time_test_total = difference.seconds
        self.solution_serializer.accuracy_per_class = self.accuracy_per_class
        self.solution_serializer.mean_total_accuracy = self.mean_total_accuracy
        # self.solution_serializer.correct_normal_predictions = (
        #     f"{normal_correct}/{len(self.test_normal_dataset)}"
        # )
        # self.solution_serializer.correct_abnormal_predictions = (
        #     f"{anomaly_correct}/{len(anomaly_dataset)}"
        # )
        ## Save run serialization data -- end

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

        full_length = [x.shape[0] for x in list_of_dataFrames]
        full_length = sum(full_length)
        self.solution_serializer.dataset_full_size = (
            full_length,
            list_of_dataFrames[0].shape[1],
        )
        try:
            train_df = list_of_dataFrames[
                list_of_dataFramesUtilityLabels.index("train")
            ]
            print("here?")
            cols = train_df.shape[1]
            print(application_instance_metadata.dataset_metadata.is_labeled)
            print(type(application_instance_metadata.dataset_metadata.is_labeled))
            if application_instance_metadata.dataset_metadata.is_labeled == True:
                print("but here?")
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

            self.solution_serializer.dataset_train_size = self.X_train.shape
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

            self.solution_serializer.dataset_test_size = self.X_test.shape
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

            self.solution_serializer.dataset_val_size = self.X_val.shape
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

            self.solution_serializer.dataset_run_size = self.X_run.shape
        except:
            info_message = "No run dataFrame"
            self.Logger.info(self, info_message)

        info_message = "adapt_dataset() -- end"
        self.Logger.info(self, info_message)

        return 0, f"{self} -- adapt_dataset() completed successfully"

    ## -------------- Particular methods --------------
    def plot_confusion_matrix(self, sklearn_confusion_matrix, classes, normalize=False):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            sklearn_confusion_matrix = (
                sklearn_confusion_matrix.astype("float")
                / sklearn_confusion_matrix.sum(axis=1)[:, np.newaxis]
            )
            info_message = "Creating confusion matrix, normalized"
            self.Logger.info(self, info_message)
        else:
            info_message = "Creating confusion matrix, non-normalized"
            self.Logger.info(self, info_message)

        plot_name = "Confusion matrix"
        plt.imshow(sklearn_confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(plot_name)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = ".2f" if normalize else "d"
        thresh = sklearn_confusion_matrix.max() / 2.0

        for i, j in itertools.product(
            range(sklearn_confusion_matrix.shape[0]),
            range(sklearn_confusion_matrix.shape[1]),
        ):
            plt.text(
                j,
                i,
                format(sklearn_confusion_matrix[i, j], fmt),
                horizontalalignment="center",
                color="white" if sklearn_confusion_matrix[i, j] > thresh else "black",
            )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        if self.app_instance_metadata.shared_definitions.plot_show_at_runtime == True:
            plt.show()

        ## Save plot section -- begin
        plot_save_filename, plot_save_location = get_plot_save_filename(
            plot_name.replace(" ", "_"), "ekg2", self.app_instance_metadata
        )
        plt.savefig(
            plot_save_location,
            format=self.app_instance_metadata.shared_definitions.plot_savefile_format,
        )
        info_message = f"Created picture at ./project_web/backend/images/ || {plot_save_location} || {plot_name}"
        self.Logger.info(self, info_message)
        self.plots_filenames.append(plot_save_location.split(os.sep)[-1])
        plt.clf()
        ## Save plot section -- end

    def print_results(self, predictions):

        print(predictions.shape)
        number_of_classes = self.app_instance_metadata.dataset_metadata.number_of_classes
        list_of_classes = self.app_instance_metadata.dataset_metadata.class_names

        list_of_discovered_rows = []
        for i in range(number_of_classes):
            temp_list = []
            list_of_discovered_rows.append(temp_list)

        columns = predictions.shape[1]
        for i in range(predictions.shape[0]):
            row_max = max(predictions[i])
            for j in range(columns):
                if predictions[i][j] == row_max:
                    list_of_discovered_rows[j].append(i)

        for i in range(number_of_classes):
            if len(list_of_classes) != 0:
                info_message = f"Found {len(list_of_discovered_rows[i])} entries in class {list_of_classes[i]}. A full list can be found..."
                self.Logger.info(self, info_message)
                # self.Logger.info(self, str(list_of_discovered_rows[i]))
            else:
                info_message = f"Found {len(list_of_discovered_rows[i])} entries in class no. {i + 1} (inside index {i}). A full list can be found..."
                self.Logger.info(self, info_message)
                # self.Logger.info(self, str(list_of_discovered_rows[i]))
