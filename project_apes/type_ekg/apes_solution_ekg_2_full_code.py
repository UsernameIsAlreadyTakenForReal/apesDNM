# Number of Samples: 109446
# Number of Categories: 5
# Sampling Frequency: 125Hz
# Data Source: Physionet's MIT-BIH Arrhythmia Dataset
# Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]


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

for dirname, _, filenames in os.walk("../../../Datasets/Dataset - ECG_Heartbeat"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


time_dataset_manipulation_begin = datetime.now()
train_df = pd.read_csv(
    "../../../Datasets/Dataset - ECG_Heartbeat/mitbih_train.csv", header=None
)  # header=None for it to pick up row 1 too
test_df = pd.read_csv("../../../Datasets/Dataset - ECG_Heartbeat/mitbih_test.csv", header=None)

train_df[187] = train_df[187].astype(int)
# how many of each class
# equilibre = train_df[187].value_counts()
# print(equilibre)


# We can underligne a huge difference in the balanced of the classes. After some try i have decided to choose the resample technique more than the class weights for the algorithms.

# plt.figure(figsize=(20, 10))
# my_circle = plt.Circle((0, 0), 0.7, color="white")
# plt.pie(
#     equilibre,
#     labels=["n", "q", "v", "s", "f"],
#     colors=["red", "green", "blue", "skyblue", "orange"],
#     autopct="%1.1f%%",
# )
# p = plt.gcf()
# p.gca().add_artist(my_circle)
# plt.show()

from sklearn.utils import resample

# df_1 = train_df[train_df[187] == 1]
# df_2 = train_df[train_df[187] == 2]
# df_3 = train_df[train_df[187] == 3]
# df_4 = train_df[train_df[187] == 4]
# df_0 = (train_df[train_df[187] == 0]).sample(n=20000, random_state=42)

# df_1_upsample = resample(df_1, replace=True, n_samples=20000, random_state=123)
# df_2_upsample = resample(df_2, replace=True, n_samples=20000, random_state=124)
# df_3_upsample = resample(df_3, replace=True, n_samples=20000, random_state=125)
# df_4_upsample = resample(df_4, replace=True, n_samples=20000, random_state=126)

# train_df = pd.concat([df_0, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample])


# equilibre = train_df[187].value_counts()
# print(equilibre)


# plt.figure(figsize=(20, 10))
# my_circle = plt.Circle((0, 0), 0.7, color="white")
# plt.pie(
#     equilibre,
#     labels=["n", "q", "v", "s", "f"],
#     colors=["red", "green", "blue", "skyblue", "orange"],
#     autopct="%1.1f%%",
# )
# p = plt.gcf()
# p.gca().add_artist(my_circle)
# plt.show()

# ##


# c = train_df.groupby(187, group_keys=False).apply(lambda train_df: train_df.sample(1))

# c

# plt.plot(c.iloc[0, :186])


# def plot_hist(class_number, size, min_, bins):
#     img = train_df.loc[train_df[187] == class_number].values
#     img = img[:, min_:size]
#     img_flatten = img.flatten()

#     final1 = np.arange(min_, size)
#     for i in range(img.shape[0] - 1):
#         tempo1 = np.arange(min_, size)
#         final1 = np.concatenate((final1, tempo1), axis=None)
#     print(len(final1))
#     print(len(img_flatten))
#     plt.hist2d(final1, img_flatten, bins=(bins, bins), cmap=plt.cm.jet)
#     plt.show()


# plot_hist(0, 70, 5, 65)

# plt.plot(c.iloc[1, :186])

# plot_hist(1, 50, 5, 45)

# plt.plot(c.iloc[2, :186])

# plot_hist(2, 50, 5, 45)

# plt.plot(c.iloc[3, :186])

# plot_hist(3, 60, 15, 45)

# plt.plot(c.iloc[4, :186])

# plot_hist(4, 50, 15, 35)


def add_gaussian_noise(signal):
    noise = np.random.normal(0, 0.5, 186)
    return signal + noise


# tempo = c.iloc[0, :186]
# bruiter = add_gaussian_noise(tempo)

# plt.subplot(2, 1, 1)
# plt.plot(c.iloc[0, :186])

# plt.subplot(2, 1, 2)
# plt.plot(bruiter)

# plt.show()

# time_dataset_manipulation = (datetime.now() - time_dataset_manipulation_begin).seconds
# print(
#     f"Took {time_dataset_manipulation} seconds to manipulate dataset (show graphs n shit)."
# )

####
time_dataset_split_begin = datetime.now()
target_train = train_df[187]
target_test = test_df[187]
y_train = to_categorical(target_train)
y_test = to_categorical(target_test)


X_train = train_df.iloc[:, :186].values
X_test = test_df.iloc[:, :186].values
for i in range(len(X_train)):
    X_train[i, :186] = add_gaussian_noise(X_train[i, :186])
X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)

time_dataset_split = (datetime.now() - time_dataset_split_begin).seconds
print(f"Took {time_dataset_split} seconds to split dataset.")


def network(X_train, y_train, X_test, y_test):
    im_shape = (X_train.shape[1], 1)
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

    model = Model(inputs=inputs_cnn, outputs=main_output)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8),
        ModelCheckpoint(
            filepath="best_model.h5", monitor="val_loss", save_best_only=True
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=40,
        callbacks=callbacks,
        batch_size=32,
        validation_data=(X_test, y_test),
    )
    model.load_weights("best_model.h5")
    return (model, history)


def evaluate_model(history, X_test, y_test, model):
    scores = model.evaluate((X_test), y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    print(history)
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model - Accuracy")
    plt.legend(["Training", "Validation"], loc="lower right")
    plt.show()

    fig2, ax_loss = plt.subplots()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model- Loss")
    plt.legend(["Training", "Validation"], loc="upper right")
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.show()
    target_names = ["0", "1", "2", "3", "4"]

    y_true = []
    for element in y_test:
        y_true.append(np.argmax(element))
    prediction_proba = model.predict(X_test)
    prediction = np.argmax(prediction_proba, axis=1)
    cnf_matrix = confusion_matrix(y_true, prediction)


time_model_begin = datetime.now()
model, history = network(X_train, y_train, X_test, y_test)
time_model = (datetime.now() - time_model_begin).seconds
print(f"Took {time_model} seconds to create model")

time_model_eval_begin = datetime.now()
evaluate_model(history, X_test, y_test, model)
time_model_eval = (datetime.now() - time_model_eval_begin).seconds
print(f"Took {time_model_eval} seconds to evaluate model.")

time_model_predict_begin = datetime.now()
y_pred = model.predict(X_test)
time_model_predict = (datetime.now() - time_model_predict_begin).seconds
print(f"Took {time_model_predict} seconds to predict with model.")


import itertools


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# Compute confusion matrix
time_confusion_matrix_begin = datetime.now()
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
np.set_printoptions(precision=2)
time_confusion_matrix = (datetime.now() - time_confusion_matrix_begin).seconds
print(f"Took {time_confusion_matrix} seconds to compute confusion matrix.")

# Plot non-normalized confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(
    cnf_matrix,
    classes=["N", "S", "V", "F", "Q"],
    normalize=True,
    title="Confusion matrix, with normalization",
)
plt.show()
