# import sys
# print(sys.path)
# sys.path.append("py36\lib\site-packages\numpy\__init__.py")
import numpy as np
import tensorflow as tf

with tf.Session() as sess:
    a = tf.constant(15, name='a')
    b = tf.constant(5, name='b')
    prod = tf.multiply(a, b, name='Multiply')
    sum = tf.add(a, b, name='Sum')
    res = tf.divide(prod, sum, name='Divide')

    out = sess.run(res)
    print(out)
