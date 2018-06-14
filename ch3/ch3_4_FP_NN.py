# -*- coding:utf-8 -*-
# 通过变量实现神经网络的参数并实现前向传播的过程
import tensorflow as tf

# 声明w1、w2两个变量。通过seed参数设定随机种子，
# 保证每次运行的结果一样
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 暂时将输入的特征向量定义为一个常量。注意这里x是一个1×2的矩阵
x = tf.constant([[0.7, 0.9]])

# 通过前向传播获得神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


sess = tf.Session()
# 由于w1和w2都还没有运行初始化过程，所以不能直接运行sess.run(y)。
# 需要先初始化w1、w2
#sess.run(w1.initializer)
#sess.run(w2.initializer)
init_op = tf.initialize_all_variables()
sess.run(init_op)
print(sess.run(y))
sess.close()

a1 = tf.Variable(tf.random_normal([2, 3], stddev=1), name="w1")
a2 = tf.Variable(tf.random_normal([2, 2], stddev=1), name="w2")
tf.assign(a1, a2, validate_shape=False)



