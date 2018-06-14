# -*- coding:utf-8 -*-
import tensorflow as tf
# tf.constant是一个计算，这个计算的结果为一个张量，保存在变量a中
a = tf.constant([1, 2], name="a", dtype=tf.float32)
b = tf.constant([2.0, 3.0], name="b")
result = tf.add(a, b, name="result")
print(result)

# 手动创建会话
sess = tf.Session()
with sess.as_default():
    print(result.eval())
