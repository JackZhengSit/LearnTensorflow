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

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

# parameter for LSTM
data_dim = 29
timesteps = 1
num_classes = 2
batch_size = 50
hide_units = 50
dropout = 0.5
learn_rate = 0.001
epochs = 20

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

class roc_callback(Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        print(self.x.shape)
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc, 4)), str(round(roc_val, 4))), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


model = Sequential()
model.add(LSTM(hide_units, return_sequences=True, dropout=dropout,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(hide_units, return_sequences=True, dropout=dropout))
model.add(LSTM(hide_units, dropout=dropout))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),
          callbacks=[TensorBoard(log_dir='logs_auc/'),
                     roc_callback(training_data=(X_train, y_train), validation_data=(X_test, y_test))])

model.save('models/FraudDetectionModel_auc.h5')

# X_train = dataset.iloc[:, 0:28].values
# y_train = dataset.iloc[:, 29].values
# X_train.reshape(shape=(-1, 29, 1))
# y_train.reshape(shape=(-1, 1))

#
