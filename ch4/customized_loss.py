# -*- coding:utf-8 -*-
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# 2个输入结点
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
# 回归问题只有1个输出结点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义一个单层的神经网络前向传播过程，只是简单加权和
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义预测多了和预测少了的成本
loss_less = 1
loss_more = 10
loss = tf.reduce_sum(
    tf.select(tf.greater(y, y_), loss_more*(y-y_), loss_less*(y_-y)))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 定义损失函数为均方误差
loss_2 = tf.reduce_mean(tf.square(y_ - y))
train_step_2 = tf.train.AdamOptimizer(0.001).minimize(loss_2)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 设置回归的正确值为两个输入加上一个随机量
Y = [[x1 + x2 + rdm.rand()/10.0-0.05] for (x1, x2) in X]

# 训练神经网络
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        #sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        sess.run(train_step_2, feed_dict={x: X[start:end], y_: Y[start:end]})
        print sess.run(w1)



