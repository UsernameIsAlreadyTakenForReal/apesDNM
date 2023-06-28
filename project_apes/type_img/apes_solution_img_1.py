# This file:
# Contains the solution_img_1. Builds a CNN using the 'building blocks' Tensorflow & Keras offer.
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


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
import json
import sys
import os

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from datetime import datetime

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


class Solution_img_1:
    def __init__(self, app_instance_metadata, Logger):
        self.Logger = Logger
        self.Logger.info(self, "Creating object of type solution_img_1")

        self.solution_serializer = Solution_Serializer()
        self.solution_serializer._app_instance_ID = (
            app_instance_metadata.app_instance_ID
        )
        self.solution_serializer._time_object_creation = datetime.now().strftime(
            "%Y-%m-%d_%H:%M:%S"
        )

        self.batch_size = 32

        self.plots_filenames = []
        # self.datasets_names = []
        self.solution_serializer.solution_name = "img1"
        self.solution_serializer.dataset_name = (
            app_instance_metadata.dataset_metadata.dataset_name_stub
        )
        self.solution_serializer.batch_size = self.batch_size

        self.app_instance_metadata = app_instance_metadata
        self.project_solution_model_filename = (
            app_instance_metadata.shared_definitions.project_solution_img1_model_filename
        )
        self.last_good_suitable_model = (
            app_instance_metadata.shared_definitions.project_solution_img1_d_img1_best_model_filename
        )
        pass

    ## -------------- To JSON --------------
    def toJSON(self):  # this is not used
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def save_run(self):
        info_message = "save_run() -- begin"
        self.Logger.info(self, info_message)

        self.solution_serializer.plots_filenames = self.plots_filenames

        write_to_solutions_runs_json_file(
            self.Logger, "img1", self.solution_serializer, self.app_instance_metadata
        )

        info_message = "save_run() -- assess_whether_to_save_model_as_best"
        self.Logger.info(self, info_message)

        # if self.solution_serializer._used_train_function == True:
        #     assess_whether_to_save_model_as_best(
        #         self.Logger,
        #         self.app_instance_metadata,
        #         "img1",
        #         self.MODEL_SAVE_PATH.split("/")[-1],
        #         self.accuracy_per_class,
        #         self.mean_total_accuracy,
        #     )

        info_message = "save_run() -- end"
        self.Logger.info(self, info_message)

        return 0, f"{self} -- save_run() completed successfully"

    ## -------------- General methods --------------
    ## Model methods
    def create_model(self):
        info_message = "create_model() -- begin"
        self.Logger.info(self, info_message)

        self.model = Sequential(
            [
                layers.Rescaling(
                    1.0 / 255, input_shape=(self.image_height, self.image_width, 3)
                ),
                layers.Conv2D(16, 3, padding="same", activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding="same", activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding="same", activation="relu"),
                layers.MaxPooling2D(),
                layers.Dropout(0.2),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(self.num_classes),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        self.model.summary()

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
                "s_img1_d_"
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
                "img1",
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
                    self.Logger, "img1", self.app_instance_metadata
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

    ## Functionality method
    def train(self, epochs=40):
        info_message = "train() -- begin"
        self.Logger.info(self, info_message)

        info_message = "##########################################################"
        self.Logger.info(self, info_message)
        info_message = (
            f"Begining training for solution_img_1. Number of epochs: {epochs}"
        )
        self.Logger.info(self, info_message)
        info_message = "##########################################################"
        self.Logger.info(self, info_message)

        f1_time = datetime.now()

        history = self.model.fit(
            self.train_ds, validation_data=self.val_ds, epochs=epochs
        )

        self.acc = history.history["accuracy"]
        self.val_acc = history.history["val_accuracy"]

        self.loss = history.history["loss"]
        self.val_loss = history.history["val_loss"]

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
        self.solution_serializer.train_acc = self.acc
        self.solution_serializer.val_acc = self.val_acc
        self.solution_serializer.train_loss = self.loss
        self.solution_serializer.val_loss = self.val_loss
        ## Save run serialization data -- end

        info_message = f"Training - it took {difference} for {epochs} epochs"
        self.Logger.info(self, info_message)

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, self.loss, label="Training Loss")
        plt.plot(epochs_range, self.val_loss, label="Validation Loss")
        plt.legend(loc="lower left")
        plt.title("Training and Validation Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, self.acc, label="Training Accuracy")
        plt.plot(epochs_range, self.val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")
        if self.app_instance_metadata.shared_definitions.plot_show_at_runtime == True:
            plt.show()

        ## Save plot section -- begin
        plot_name = "Training and Validation Accuracy" + " -- " + "Training and Validation Loss"
        plot_save_filename, plot_save_location = get_plot_save_filename(
            plot_name.replace(" ", "_"), "img1", self.app_instance_metadata
        )
        plt.savefig(
            plot_save_location,
            format=self.app_instance_metadata.shared_definitions.plot_savefile_format,
        )
        info_message = f"Created picture at ./project_web/backend/images/ || {plot_save_location} || {plot_name}"
        self.Logger.info(self, info_message)
        self.plots_filenames.append(plot_save_location.split(os.sep)[-1])
        ## Save plot section -- end


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
        info_message = f"Begining testing for solution_img_1"
        self.Logger.info(self, info_message)
        info_message = "##########################################################"
        self.Logger.info(self, info_message)

        return 0, f"{self} -- test() completed successfully"

    ## Dataset methods
    def adapt_dataset(
        self,
        application_instance_metadata,
        images_data_directory,
        images_information,
    ):
        info_message = "adapt_dataset() -- begin"
        self.Logger.info(self, info_message)

        keras_images_path = pathlib.Path(images_data_directory)

        self.image_height = images_information["height"]
        self.image_width = images_information["width"]

        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            keras_images_path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.image_height, self.image_width),
            batch_size=self.batch_size,
        )

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            keras_images_path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.image_height, self.image_width),
            batch_size=self.batch_size,
        )

        self.class_names = self.train_ds.class_names
        self.num_classes = len(self.class_names)

        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = (
            self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        )
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Get shapes of tensors
        for image_batch, labels_batch in self.train_ds:
            print(image_batch.shape)
            print(labels_batch.shape)
            break

        # Save serialization data -- begin
        self.solution_serializer.class_names = self.class_names
        self.solution_serializer.num_classes = self.num_classes
        self.solution_serializer.tensor_shape_image_shape = image_batch
        self.solution_serializer.tensor_shape_labels_batch = labels_batch
        # Save serialization data -- end

        info_message = "adapt_dataset() -- end"
        self.Logger.info(self, info_message)

        return 0, f"{self} -- adapt_dataset() completed successfully"
