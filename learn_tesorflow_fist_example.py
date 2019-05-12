import tensorflow as tf
import numpy as np

# crearte data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.2 + 0.3

# construct model
weights = tf.Variable(tf.random_uniform([1], -1, 1))
biases = tf.Variable(tf.zeros([1]))

y = weights * x_data + biases

# define loss
loss = tf.reduce_mean(tf.square(y - y_data))

# optimize loss
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# init all Variable
init = tf.global_variables_initializer()

# Create session to train
sess = tf.Session()
sess.run(init)

# train and print
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(weights), sess.run(biases))
