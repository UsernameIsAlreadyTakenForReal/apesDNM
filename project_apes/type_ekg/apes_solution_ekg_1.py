# This file:
# Contains the solution_ekg_1. Builds a Long Short-Time-Memory Neural Network using an Encoder-Decoder using Torch.
# This solution only searches for normal vs. anomaly, does not interpret each class
#
# Main methods to be called from apes_application:
# create_model(), load_model()
#       -- purpose: self.model exists
# save_model()
#       -- purpose: self.model save
# train(epochs), test()
#       -- purpose: train and use self.model
# adapt_dataset(self, application_instance_metadata, list_of_dataFrames, list_of_dataFramesUtilityLabels)
#       -- purpose: self.train_dataset, self.val_dataset, self.test_normal_dataset, self.test_anomaly_dataset etc exist

import arff
import os, psutil
import copy
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from matplotlib import rc
from sklearn.model_selection import train_test_split
from pylab import rcParams
from datetime import datetime
from sklearn import metrics

import torch
from torch import nn, optim
import torch.nn.functional as F

torch.__version__


import sys

sys.path.insert(1, "../helpers_aiders_and_conveniencers")
from helpers_aiders_and_conveniencers.misc_functions import (
    get_last_model,
    model_filename_fits_expected_name,
    get_full_path_of_given_model,
    get_plot_save_filename,
    write_to_solutions_runs_json_file,
)
from helpers_aiders_and_conveniencers.solution_serializer import Solution_Serializer


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)

        return hidden_n.reshape((self.n_features, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim, hidden_size=input_dim, num_layers=1, batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))

        return self.output_layer(x)


class Recurrent_Autoencoder(nn.Module):
    def __init__(self, device, seq_len, n_features, embedding_dim=64):
        super(Recurrent_Autoencoder, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class Solution_ekg_1:
    def __init__(self, app_instance_metadata, Logger):
        self.Logger = Logger
        self.Logger.info(self, "Creating object of type solution_ekg_1")

        self.solution_serializer = Solution_Serializer()
        self.solution_serializer._app_instance_ID = (
            app_instance_metadata.app_instance_ID
        )
        self.solution_serializer._time_object_creation = datetime.now().strftime(
            "%Y-%m-%d_%H:%M:%S"
        )

        self.plots_filenames = []
        # self.datasets_names = []
        self.solution_serializer.solution_name = "ekg1"
        self.solution_serializer.dataset_name = (
            app_instance_metadata.dataset_metadata.dataset_name_stub
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.solution_serializer.device_used = str(self.device)
        info_message = f"Using device {self.device}"
        self.Logger.info(self, info_message)

        self.app_instance_metadata = app_instance_metadata
        self.project_solution_model_filename = (
            app_instance_metadata.shared_definitions.project_solution_ekg1_model_filename
        )
        self.last_good_suitable_model = ""
        match self.solution_serializer.dataset_name:
            case "ekg1":
                self.last_good_suitable_model = (
                    app_instance_metadata.shared_definitions.project_solution_ekg1_d_ekg1_best_model_filename
                )
            case "ekg2":
                self.last_good_suitable_model = (
                    app_instance_metadata.shared_definitions.project_solution_ekg1_d_ekg2_best_model_filename
                )

    ## -------------- To JSON --------------
    def toJSON(self):  # this is not used
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def save_run(self):
        info_message = "save_run() -- begin"
        self.Logger.info(self, info_message)

        self.solution_serializer.plots_filenames = self.plots_filenames

        write_to_solutions_runs_json_file(
            self.Logger, "ekg1", self.solution_serializer, self.app_instance_metadata
        )

        info_message = "save_run() -- end"
        self.Logger.info(self, info_message)

        return 0, f"{self} -- save_run() completed successfully"

    ## -------------- General methods --------------
    ## Model methods
    def create_model(self):
        info_message = "create_model() -- begin autoencoder model creation"
        self.Logger.info(self, info_message)

        self.model = Recurrent_Autoencoder(
            self.device, self.seq_len, self.n_features, 128
        )
        self.model = self.model.to(self.device)

        info_message = "create_model() -- ended autoencoder model creation"
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
            # # this could work very nicely but torch.save is a piece of shit. Instead we dump it in /project_apes
            # # TODO: find file and move it. Save it with timestamp only if it's good
            # MODEL_SAVE_PATH = (
            #     self.shared_definitions.ROOT_DIR
            #     + "\\type_ekg\\"
            #     + self.project_solution_model_filename
            # )
            MODEL_SAVE_PATH = (
                "s_ekg1_d_"
                + self.app_instance_metadata.dataset_metadata.dataset_name_stub
            )

        MODEL_SAVE_PATH = (
            MODEL_SAVE_PATH + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pth"
        )
        # Margareta = get_full_path_of_given_model(
        #     MODEL_SAVE_PATH,
        #     self.app_instance_metadata.shared_definitions.project_model_root_path,
        # )
        info_message = "Saving model at " + MODEL_SAVE_PATH
        self.Logger.info(self, info_message)
        torch.save(self.model, MODEL_SAVE_PATH)

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
            if (
                model_filename_fits_expected_name(
                    "ekg1",
                    self.app_instance_metadata,
                    self.last_good_suitable_model,
                )
                == True
            ):
                info_message = "Loading the last good model saved"
                self.Logger.info(self, info_message)
                model_absolute_path = get_full_path_of_given_model(
                    self.last_good_suitable_model,
                    self.app_instance_metadata.shared_definitions.project_model_root_path,
                )
                info_message = f"Loading {model_absolute_path}"
                self.Logger.info(self, info_message)
                self.model = torch.load(
                    model_absolute_path,
                    map_location=lambda storage, loc: storage.cuda(0),
                )
                self.solution_serializer.model_filename = model_absolute_path.split(
                    "/"
                )[-1]
            else:
                info_message = "Loading last available model"
                self.Logger.info(self, info_message)
                return_code, return_message, model_path = get_last_model(
                    self.Logger, "ekg1", self.app_instance_metadata
                )
                if return_code != 0:
                    self.Logger.info(self, return_message)
                    return return_code, return_message
                else:
                    info_message = f"Loading {model_path}"
                    self.Logger.info(self, info_message)
                    self.model = torch.load(
                        model_path, map_location=lambda storage, loc: storage.cuda(0)
                    )
                    self.solution_serializer.model_filename = model_path.split("/")[-1]

        info_message = "load_model() -- end"
        self.Logger.info(self, info_message)

        return 0, f"{self} -- load_model() completed successfully"

    ## Functionality methods
    def train(self, epochs=40):
        info_message = "train() -- begin"
        self.Logger.info(self, info_message)

        f1_time = datetime.now()

        info_message = "##########################################################"
        self.Logger.info(self, info_message)
        info_message = f"Beginning training for solution_ekg_1. Number of epochs: {epochs}. Using device {self.device}"
        self.Logger.info(self, info_message)
        info_message = "##########################################################"
        self.Logger.info(self, info_message)

        model, history = self.train_model_helper(
            self.model,
            self.train_dataset,
            self.val_dataset,
            n_epochs=epochs,
        )
        self.model = model

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

        ax = plt.figure().gca()
        ax.plot(history["train"])
        ax.plot(history["val"])
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["train", "test"])
        plot_name = "Loss over training epochs"
        plt.title(plot_name)
        if self.app_instance_metadata.shared_definitions.plot_show_at_runtime == True:
            plt.show()

        ## Save plot section -- begin
        plot_save_filename, plot_save_location = get_plot_save_filename(
            plot_name.replace(" ", "_"), "ekg1", self.app_instance_metadata
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

        info_message = "Training - it took {time} for {number} epochs".format(
            time=difference, number=epochs
        )
        self.Logger.info(self, info_message)
        info_message = "train() -- end"
        self.Logger.info(self, info_message)

        return 0, f"{self} -- train() completed successfully"

    def test(self):
        info_message = "test() -- begin"
        self.Logger.info(self, info_message)

        f1_time = datetime.now()

        info_message = "##########################################################"
        self.Logger.info(self, info_message)
        info_message = (
            f"Beginning testing for solution_ekg_1. Using device {self.device}"
        )
        self.Logger.info(self, info_message)
        info_message = "##########################################################"
        self.Logger.info(self, info_message)

        _, losses = self.predict(self.model, self.train_dataset)
        sns.displot(losses, bins=50, kde=True)

        THRESHOLD = 46

        # normal heartbeats
        predictions, pred_losses = self.predict(self.model, self.test_normal_dataset)
        print(len(predictions))
        print(len(predictions[0]))
        sns.displot(pred_losses, bins=50, kde=True)
        normal_correct = sum(l <= THRESHOLD for l in pred_losses)
        info_message = f"Correct normal predictions: {normal_correct}/{len(self.test_normal_dataset)}"
        self.Logger.info(self, info_message)

        #self.print_results(predictions)

        # anomalies
        anomaly_dataset = self.test_anomaly_dataset[: len(self.test_normal_dataset)]
        predictions, pred_losses = self.predict(self.model, anomaly_dataset)
        #print(predictions)
        sns.displot(pred_losses, bins=50, kde=True)
        anomaly_correct = sum(l > THRESHOLD for l in pred_losses)
        info_message = (
            f"Correct anomaly predictions: {anomaly_correct}/{len(anomaly_dataset)}"
        )
        self.Logger.info(self, info_message)

        #self.print_results(predictions)

        f2_time = datetime.now()
        difference = f2_time - f1_time
        seconds_in_day = 24 * 60 * 60
        divmod(difference.days * seconds_in_day + difference.seconds, 60)

        ## Save run serialization data -- begin
        self.solution_serializer.time_test_start = f1_time.strftime("%Y-%m-%d_%H:%M:%S")
        self.solution_serializer._used_test_function = True
        self.solution_serializer.time_test_end = f2_time.strftime("%Y-%m-%d_%H:%M:%S")
        self.solution_serializer.time_test_total = difference.seconds
        self.solution_serializer.correct_normal_predictions = (
            f"{normal_correct}/{len(self.test_normal_dataset)}"
        )
        self.solution_serializer.correct_abnormal_predictions = (
            f"{anomaly_correct}/{len(anomaly_dataset)}"
        )
        self.solution_serializer.test_loss_average = sum(pred_losses) / len(pred_losses)
        self.solution_serializer.prediction_threshold = THRESHOLD
        ## Save run serialization data -- end

        plt.figure()
        plt.plot(self.test_normal_dataset[0])
        plt.plot(predictions[0])
        plt.legend(["beat", "prediction"])
        plot_name = "Beat and prediction"
        plt.title(plot_name)
        if self.app_instance_metadata.shared_definitions.plot_show_at_runtime == True:
            plt.show()

        ## Save plot section -- begin
        plot_save_filename, plot_save_location = get_plot_save_filename(
            plot_name.replace(" ", "_"), "ekg1", self.app_instance_metadata
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

        info_message = "test() -- end"
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

        RANDOM_SEED = 42

        df = pd.concat(list_of_dataFrames)
        new_columns = list(df.columns)
        new_columns[-1] = "target"
        df.columns = new_columns

        class_normal = (
            application_instance_metadata.dataset_metadata.numerical_value_of_desired_label
        )
        if class_normal != 0:
            return 1, f"{self} adapt_dataset() -- class_normal == {class_normal}"

        normal_df = df[df.target == int(class_normal)].drop(labels="target", axis=1)
        self.anormal_df = df[df.target != int(class_normal)].drop(
            labels="target", axis=1
        )

        self.train_df, val_df = train_test_split(
            normal_df, test_size=0.15, random_state=RANDOM_SEED
        )
        info_message = "Created train_df"
        self.Logger.info(self, info_message)

        self.val_df, self.test_df = train_test_split(
            val_df, test_size=0.33, random_state=RANDOM_SEED
        )
        info_message = "Created val_df, test_df"
        self.Logger.info(self, info_message)

        self.train_dataset, self.seq_len, self.n_features = self.create_dataset(
            self.train_df
        )
        self.val_dataset, _, _ = self.create_dataset(self.val_df)
        self.test_normal_dataset, _, _ = self.create_dataset(self.test_df)
        self.test_anomaly_dataset, _, _ = self.create_dataset(self.anormal_df)

        self.solution_serializer.dataset_train_size = self.train_df.shape
        self.solution_serializer.dataset_test_size = self.test_df.shape
        self.solution_serializer.dataset_val_size = self.val_df.shape
        self.solution_serializer.dataset_full_size = df.shape

        return 0, f"{self} -- adapt_dataset() completed successfully"

    ## -------------- Particular methods --------------
    def train_model_helper(self, model, train_dataset, val_dataset, n_epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.L1Loss(reduction="sum").to(self.device)
        history = dict(train=[], val=[])

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 10000.0

        for epoch in range(1, n_epochs + 1):
            time_temp = datetime.now()
            info_message = f"Epoch {epoch}: start"
            self.Logger.info(self, info_message)
            model = model.train()

            train_losses = []
            for seq_true in train_dataset:
                optimizer.zero_grad()

                seq_true = seq_true.to(self.device)
                seq_pred = model(seq_true)

                loss = criterion(seq_pred, seq_true)

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            val_losses = []
            model = model.eval()
            with torch.no_grad():
                for seq_true in val_dataset:
                    seq_true = seq_true.to(self.device)
                    seq_pred = model(seq_true)

                    loss = criterion(seq_pred, seq_true)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            history["train"].append(train_loss)
            history["val"].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            info_message = f"Epoch {epoch}: train loss {train_loss}%, val loss {val_loss}%. Time: {(datetime.now() - time_temp).seconds}s"
            self.Logger.info(self, info_message)

        self.solution_serializer.train_loss = train_loss
        self.solution_serializer.train_val_loss = val_loss

        model.load_state_dict(best_model_wts)
        return model.eval(), history

    def predict(self, model, dataset):
        predictions, losses = [], []
        criterion = nn.L1Loss(reduction="sum").to(self.device)
        with torch.no_grad():
            model = model.eval()
            for seq_true in dataset:
                seq_true = seq_true.to(self.device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                predictions.append(seq_pred.cpu().numpy().flatten())
                losses.append(loss.item())
        return predictions, losses

    def confusion_matrix(self):
        actual = np.random.binomial(1, 0.9, size=150)
        predicted = np.random.binomial(1, 0.9, size=150)

        confusion_matrix = metrics.confusion_matrix(actual, predicted)

        cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix, display_labels=["Anomaly", "Normal"]
        )

        cm_display.plot()
        if self.app_instance_metadata.shared_definitions.plot_show_at_runtime == True:
            plt.show()

        plot_name = "Confusion matrix"
        ## Save plot section -- begin
        plot_save_filename = get_plot_save_filename(
            plot_name.replace(" ", "_"), "ekg1", self.app_instance_metadata
        )
        plot_save_location = (
            os.path.abspath(
                self.app_instance_metadata.shared_definitions.plot_savefile_location
            )
            + os.sep
            + plot_save_filename
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

    def create_dataset(self, df):
        sequences = df.astype(np.float32).to_numpy().tolist()
        dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
        n_seq, seq_len, n_features = torch.stack(dataset).shape
        return dataset, seq_len, n_features

    def print_results(self, predictions):

        print(predictions.shape)
        number_of_classes = 2
        list_of_classes = ["normal", "anormal"]

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