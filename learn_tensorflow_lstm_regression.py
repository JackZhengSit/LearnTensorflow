import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# parameters
BATCH_START = 0  # batch started index
TIME_STEPS = 20  # the step of backpropagation through time
BATCH_SIZE = 50  #
INPUT_SIZE = 1  # sin input size
OUTPUT_SIZE = 1  # cos output size
CELL_SIZE = 10  # size of hidden unit
LR = 0.006  # learning rate

class LSTM
