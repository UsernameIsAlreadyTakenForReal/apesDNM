import arff
import torch

import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
import os, psutil
from datetime import datetime

from torch import nn, optim

import torch.nn.functional as F


## This one is for anomaly_detection following the tutorial:
## https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
## For each instruction in the tutorial, I included the comment above it, or something similar, so one can
## ctrl+f and find it easily on the website, since the website uses some shit we do not posses

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
## --------------------------- Getting the datasets ---------------------------
###############################################################################

## Train data (500 entries)
file_path = "../../Dataset - ECG5000/ECG5000_TRAIN.arff"
data = arff.load(file_path)

train_list = []
train_class_list = []
train_number_of_entries = 0

for row in data:
    train_number_of_entries = train_number_of_entries + 1
    temp_row = []
    for i in range(0, 140):
        temp_row.append(row[i])
    train_class_list.append(int(row.target))
    train_list.append(temp_row)
    

train_data = np.zeros((train_number_of_entries, 140))
train_labels = np.array(train_class_list)
 
for i in range(0, train_number_of_entries):
    something_normal_for_once_thank_you = [float(el) for el in train_list[i]]
    train_data[i] = np.array(something_normal_for_once_thank_you)
    
all_train_data = np.c_[train_data, train_labels]



## Test data (4500 entries)
file_path = "../../Dataset - ECG5000/ECG5000_TEST.arff"
data = arff.load(file_path)

test_list = []
test_class_list = []
test_number_of_entries = 0

for row in data:
    test_number_of_entries = test_number_of_entries + 1
    temp_row = []
    for i in range(0, 140):
        temp_row.append(row[i])
    test_class_list.append(int(row.target))
    test_list.append(temp_row)
    

test_data = np.zeros((test_number_of_entries, 140))
test_labels = np.array(test_class_list)
 
for i in range(0, test_number_of_entries):
    something_normal_for_once_thank_you = [float(el) for el in test_list[i]]
    test_data[i] = np.array(something_normal_for_once_thank_you)
    
all_test_data = np.c_[test_data, test_labels]

###############################################################################
## ----------------------- Converting datasets to pandas ----------------------
###############################################################################
df = pd.DataFrame(all_train_data)
df_2 = pd.DataFrame(all_test_data)

df = df.append(df_2)

CLASS_NORMAL = 1
class_names = ['Normal','R on T','PVC','SP','UB']

new_columns = list(df.columns)
new_columns[-1] = 'target'
df.columns = new_columns


## Let’s get all normal heartbeats and drop the target (class) column:
normal_df = df[df.target == int(CLASS_NORMAL)].drop(labels='target', axis=1)

## We’ll merge all other classes and mark them as anomalies:
anomaly_df = df[df.target != int(CLASS_NORMAL)].drop(labels='target', axis=1)

## We’ll split the normal examples into train, validation and test sets:
train_df, val_df = train_test_split(
  normal_df,
  test_size=0.15,
  random_state=RANDOM_SEED
)

val_df, test_df = train_test_split(
  val_df,
  test_size=0.33,
  random_state=RANDOM_SEED
)

###############################################################################
## ----------------------- Definition of classes and fct ----------------------
###############################################################################

## We need to convert our examples into tensors, so we can use them to train our Autoencoder. 
## Let’s write a helper function for that:
    
def create_dataset(df):

    print("begin create_dataset")
    sequences = df.astype(np.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features


train_dataset, seq_len, n_features = create_dataset(train_df)
val_dataset, _, _ = create_dataset(val_df)
test_normal_dataset, _, _ = create_dataset(test_df)
test_anomaly_dataset, _, _ = create_dataset(anomaly_df)


## LTSM Autoencoder

## We'll use the LSTM Autoencoder from this GitHub repo with some small tweaks. 
## Our model's job is to reconstruct Time Series data. Let's start with the Encoder:
class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
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
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
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


## Our Autoencoder passes the input through the Encoder and Decoder. Let's create an instance of it:
print("begin autoencoder model definitions")
model = RecurrentAutoencoder(seq_len, n_features, 128)
model = model.to(device)


###############################################################################
## ----------------------- Train model chu chuuuuuuuuuuu ----------------------
###############################################################################
def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
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

      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:

        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)

        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
    time_per_epoch = (datetime.now() - time_temp).seconds
    print(time_per_epoch)

  model.load_state_dict(best_model_wts)
  return model.eval(), history


## We'll get the version of the model with the smallest validation error. Let's do some training: 
model, history = train_model(
  model, 
  train_dataset, 
  val_dataset, 
  n_epochs=150
)


ax = plt.figure().gca()
ax.plot(history['train'])
ax.plot(history['val'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Loss over training epochs')
plt.show();

## Let's store the model for later use:
MODEL_PATH = 'model.pth'
torch.save(model, MODEL_PATH)