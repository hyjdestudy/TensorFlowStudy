# -*- coding:utf-8 -*-
# 使用placeholder
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义placeholder作为存放输入数据的地方，不一定指定维度
# 若维度确定，给出维度可以降低出错的概率
x = tf.placeholder(tf.float32, name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
init_op = tf.initialize_all_variables()
sess.run(init_op)

print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))








