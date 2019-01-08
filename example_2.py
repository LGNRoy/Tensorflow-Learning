import tensorflow as tf
import numpy as np

from utils import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每张图片的分辨率是28×28，
# 所以我们的训练网络输入应该是28×28=784个像素数据
xs = tf.placeholder(tf.float32, [None, 784]) 
# 每张图片都表示一个数字，
# 所以我们的输出是数字0到9，共10类。
ys = tf.placeholder(tf.float32, [None, 10]) 
# 调用add_layer函数搭建一个最简单的训练网络结构，只有输入层和输出层。
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# loss函数（即最优化目标函数）选用交叉熵函数。
# 交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                            reduction_indices=[1])) # loss

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

for i in range(550):
    # 现在开始train，每次只取100张图片，免得数据太多训练太慢。
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))


