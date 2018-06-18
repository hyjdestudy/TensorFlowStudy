# -*- coding:utf-8 -*-
'''
要最小化y=x^2，初始点x0=5
'''
import tensorflow as tf

# 使用指数衰减的学习率，在迭代初期取得较高的下降速度，可以在较小的训练轮数下取得不错的收敛程度
TOTAL_ITERATIONS = 100
global_iteration = tf.Variable(0)
LEARNING_RATE = tf.train.exponential_decay(0.1,global_iteration, 1, 0.96, staircase=True)

x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y, global_step=global_iteration)

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    for current_iteration in range(TOTAL_ITERATIONS):
        sess.run(train_op)
        if current_iteration % 10 == 0:
            LEARNING_RATE_value = sess.run(LEARNING_RATE)
            x_value = sess.run(x)
            print("After %s iterations, x%s is %f, learning rate is %f."
                  % (current_iteration+1, current_iteration+1, x_value, LEARNING_RATE_value))


'''
# 学习率为0.001时，下降速度过慢
# 总的iteration次数
TRAINING_STEPS = 1000
LEARNING_RATE = 0.001

x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
loss = tf.square(x)
# 训练和优化操作
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    for i in range(TRAINING_STEPS):
        sess.run(train_op)
        if i % 100 == 0:
            x_value = sess.run(x)
            print("After %s iterations, x%s is %f." % (i+1, i+1, x_value))
'''



'''
# 学习率为1时，x在5和-5之间震荡
LEARNING_RATE = 1
TRAINING_STEPS = 10

x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
loss = tf.square(x)
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    for i in range(TRAINING_STEPS):
        sess.run(train_op)
        x_value = sess.run(x)
        print("After %s iterations, x%s is %f." % (i+1, i+1, x_value))
'''






