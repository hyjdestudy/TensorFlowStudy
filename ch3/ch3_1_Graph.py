# -*- coding:utf-8 -*-
# 测试tf的计算图，不同计算图上的张量和运算都不会共享
import tensorflow as tf

# tensorflow支持通过tf.Graph函数来生成新的计算图
g1 = tf.Graph()
with g1.as_default():
    # 在计算图g1中定义变量v，并设置初始值为0
    v = tf.get_variable("v", initializer=tf.zeros_initializer(shape=[1]))

g2 = tf.Graph()
with g2.as_default():
    # 在计算图g2中定义变量v，并设置初始值为1
    v = tf.get_variable("v", initializer=tf.ones_initializer(shape=[1]))

# 在计算图g1中读取变量v的取值
with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        # 在计算图g1中，变量v的取值应该为0，所以下面的输出为[0.]
        print(sess.run(tf.get_variable("v")))

# 在计算图g2中读取变量v的取值
with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        # 计算图g2中，变量v的取值应该为1，所以下面这行会输出[1.]
        print(sess.run(tf.get_variable("v")))


