import tensorflow as tf
import numpy as np

from utils import *

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

print_nparray("X_data", x_data, 5)
print_nparray("y_data", y_data, 5)

# 这个地方的weight和biases就是神经网络要学的东西
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

# 这里规定规则
y = weights*x_data + biases

# 计算y与y_data的误差 (MSE)
loss = tf.reduce_mean(tf.square(y-y_data))

# 用梯度下降来进行误差传递，optimizer会进行参数的更新
# 需要考虑一下梯度下降里面的参数的意义
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#初始化定义所有的变量
init = tf.global_variables_initializer()  

#创建会话
sess = tf.Session()
sess.run(init)          # Very important

print("step, weights      , biases")
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print("{0: >4}, {1}, {2}".format(step, sess.run(weights), sess.run(biases)))
        # print(step, sess.run(weights), sess.run(biases))
# 有个问题
# 如何在每个step打印loss