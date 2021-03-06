import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_inputs = 28  # MNIST data input(img shape: 28*28)row
n_steps = 28  # col
n_hidden_units = 128  # neural units in hidden layer
n_classes = 10  # MNIST classes (0-9 digits)

# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# define weights and biases
weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),  # (28,128)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))  # (128,10)
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),  # (128,)
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))  # (10,)
}


def RNN(X, weight, biases):
    # hidden layer for input to cell
    ##############################################################################################
    # X.shape(128batch, 28step, 28input)
    X = tf.reshape(X, [-1, n_inputs])  # X.shape(128*28, 28)
    X_in = tf.matmul(X, weights['in']) + biases['in']  # X_in.shape(128*28, 128)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])  # x_in.shape(128, 28, 128)

    # cell
    ##############################################################################################
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)

    # lstm cell state is divided in two parts(c_state, m_state) as a tuple
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    # hidden layer for output as the final result
    ##############################################################################################
    # method 1
    result = tf.matmul(states[1], weights['out']) + biases['out']

    # method 2
    # unpack to list[(batch, outputs)..]*steps
    # outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # state is the last outputs
    # result = tf.matmul(outputs[-1], weight['out']) + biases['out']

    return result


# train
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x: batch_xs,
                                        y: batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x: batch_xs,
                                                y: batch_ys}))
        step += 1
