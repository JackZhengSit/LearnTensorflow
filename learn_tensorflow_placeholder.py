import tensorflow as tf
import numpy as np


input1 = tf.placeholder(tf.float32,[3,1])
input2 = tf.placeholder(tf.float32,[1,3])

output=tf.matmul(input1,input2)
matrix1=np.array([[1],
                  [2],
                  [3]])
matrix2=np.array([[1,2,3]])

print(matrix1.shape,matrix2.shape)
with tf.Session() as sess:
    # print(sess.run(output,feed_dict={input1:[[1],
    #                                          [2],
    #                                          [3]],
    #                                  input2:[[1],[2],[3]]}))
    print (sess.run(output,feed_dict={input1:matrix1,input2:matrix2}))