import arff
import os, psutil
import copy
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


## Next, we'll decode the compressed representation using a Decoder:
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


## Time to wrap everything into an easy to use module:
class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class solution_ekg_1:
    def __init__(self, shared_definitions, Logger):
        self.Logger = Logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Logger.info("Creating object of type solution_ekg_1")
        self.project_solution_model_filename = (
            shared_definitions.project_solution_ekg_1_model_filename
        )
        self.project_solution_training_script = (
            shared_definitions.project_solution_ekg_1_training_script
        )

    def train_model_helper(self, model, train_dataset, val_dataset, n_epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.L1Loss(reduction="sum").to(self.device)
        history = dict(train=[], val=[])

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 10000.0

        for epoch in range(1, n_epochs + 1):
            time_temp = datetime.now()
            print("begin train_model epoch {i}".format(i=epoch))
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

            print(f"Epoch {epoch}: train loss {train_loss} val loss {val_loss}")
            time_per_epoch = (datetime.now() - time_temp).seconds
            print(time_per_epoch)

        model.load_state_dict(best_model_wts)
        return model.eval(), history

    def train(self, epochs):
        f1_time = datetime.now()
        number_of_epochs = epochs

        model, history = self.train_model_helper(
            model,
            self.train_dataset,
            self.val_dataset,
            n_epochs=number_of_epochs,
        )

        f2_time = datetime.now()
        difference = f2_time - f1_time
        seconds_in_day = 24 * 60 * 60
        divmod(difference.days * seconds_in_day + difference.seconds, 60)
        self.device

        ax = plt.figure().gca()
        ax.plot(history["train"])
        ax.plot(history["val"])
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["train", "test"])
        plt.title("Loss over training epochs")
        plt.show()

        ## Let's store the model for later use:
        MODEL_PATH = self.project_solution_model_filename
        torch.save(model, MODEL_PATH)

        print(
            "It took {time} for {number} epochs".format(
                time=difference, number=number_of_epochs
            )
        )

    def predict(model, dataset):
        predictions, losses = [], []
        criterion = nn.L1Loss(reduction="sum").to(self.device)
        with torch.no_grad():
            model = model.eval()
            for seq_true in dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                predictions.append(seq_pred.cpu().numpy().flatten())
                losses.append(loss.item())
        return predictions, losses

    def test(self, model, train_dataset):
        _, losses = self.predict(model, train_dataset)
        sns.displot(losses, bins=50, kde=True)

        # Using the threshold, we can turn the problem into a simple binary classification task:
        # If the reconstruction loss for an example is below the threshold, we’ll classify it
        #   as a normal heartbeat
        # Alternatively, if the loss is higher than the threshold, we’ll classify it as an anomaly
        THRESHOLD = 26

        # normal heartbeats
        predictions, pred_losses = predict(model, test_normal_dataset)
        sns.displot(pred_losses, bins=50, kde=True)
        correct = sum(l <= THRESHOLD for l in pred_losses)
        print(f"Correct normal predictions: {correct}/{len(test_normal_dataset)}")

        # anomalies
        anomaly_dataset = test_anomaly_dataset[: len(test_normal_dataset)]
        predictions, pred_losses = predict(model, anomaly_dataset)
        sns.displot(pred_losses, bins=50, kde=True)
        correct = sum(l > THRESHOLD for l in pred_losses)
        print(f"Correct anomaly predictions: {correct}/{len(anomaly_dataset)}")

        plt.figure(1)
        plt.plot(test_normal_dataset[0])
        plt.plot(predictions[0])
        plt.legend(["beat", "prediction"])
        plt.title("Beat and prediction")
        plt.show()

    def confusion_matrix(self):
        actual = np.random.binomial(1, 0.9, size=150)
        predicted = np.random.binomial(1, 0.9, size=150)

        confusion_matrix = metrics.confusion_matrix(actual, predicted)

        cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix, display_labels=["Anomaly", "Normal"]
        )

        cm_display.plot()
        plt.show()

    def create_model(self, seq_len, n_features):
        print("begin autoencoder model definitions")
        model = RecurrentAutoencoder(seq_len, n_features, 128)
        model = model.to(self.device)
