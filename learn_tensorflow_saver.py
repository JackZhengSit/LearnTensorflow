import tensorflow as tf
import numpy as np


# Save to file
# remember to define the same dtype and shape when restore
# W = tf.Variable([[1, 2, 3],
#                  [4, 5, 6]], dtype=tf.float32, name='weight')
# b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess, 'my_net/save_net.ckpt')
#     print('save to path', save_path)


# restore variables
# redefine the same shape and same type for your variables
W = tf.Variable(np.arange(6).reshape([2, 3]), dtype=tf.float32, name='weight')
b = tf.Variable(np.arange(3).reshape([1, 3]), dtype=tf.float32, name='biases')

# do not need init step

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'my_net/save_net.ckpt')
    print('weight:', sess.run(W))
    print('biases', sess.run(b))


