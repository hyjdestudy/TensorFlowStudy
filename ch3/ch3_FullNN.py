# -*- coding:utf-8 -*-
# 用随机数据训练一个二分类器

import tensorflow as tf
from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8

# 定义神经网络的参数，这里还是全连接+前向传播
# 输入层2个结点，隐藏层3个结点，输出层1个结点
# 所以x=[[x1, x2]], w1=[[w11, w12, w13],[w21, w22, w23]], w2=[[w1, w2, w3]]
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# placeholder,输入x和正确答案y_
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义神经网络前向传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播的算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 定义：x1+x2<1为正样本，其余为负样本。正样本标1,负样本标0
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 创建一个会话来运行tensorflow程序
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # 输出初始权重
    print "w1 = ", sess.run(w1)
    print "w2 = ", sess.run(w2)

    # 设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        # 每隔一段时间计算在所有数据上的交叉熵并输出
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training steps, cross entropy on all data is %g" % (i, total_cross_entropy))

    # 训练结束后输出参数矩阵w1、w2
    print "w1 = ", sess.run(w1)
    print "w2 = ", sess.run(w2)










