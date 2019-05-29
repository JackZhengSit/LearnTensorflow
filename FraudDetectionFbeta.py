import tensorflow as tf
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import utils
from keras.callbacks import TensorBoard
import numpy as np
from keras import backend as K


def fbeta(y_true, y_pred, beta=5):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    beta_squre = beta ** 2
    # return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return ((1 + beta_squre) * precision * recall) / (beta_squre * precision + recall + K.epsilon())


# parameter for LSTM
data_dim = 29
timesteps = 1
num_classes = 2
batch_size = 50
hide_units = 50
dropout = 0.5
learn_rate = 0.001
epochs = 50

# load data
dataset = read_csv('data/credictcard.csv', index_col=0)
dataset = dataset.iloc[0:280000]
print(dataset.describe())
print(dataset.head())
X = dataset.iloc[:, 0:29]
y = dataset.iloc[:, 29]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
X_train = np.expand_dims(X_train, 1)
X_test = np.expand_dims(X_test, 1)
# print(X_train.shape)


model = Sequential()
model.add(LSTM(hide_units, return_sequences=True, dropout=dropout,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(hide_units, return_sequences=True, dropout=dropout))
model.add(LSTM(hide_units, dropout=dropout))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[fbeta])

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),
          callbacks=[TensorBoard(log_dir='logs_fbeta/')])

model.save('models/FraudDetectionModel_fbeta.h5')

# X_train = dataset.iloc[:, 0:28].values
# y_train = dataset.iloc[:, 29].values
# X_train.reshape(shape=(-1, 29, 1))
# y_train.reshape(shape=(-1, 1))

#
