import tensorflow as tf
import numpy as np

from utils import *

# 四个输入：输入，输入大小，输出大小，激励函数
# 这里的大小值指的是列数
# 定义了一层里面的数据处理方法（主要是激励函数）
def add_layer(inputs, in_size, out_size, activation_function=None):    
    # 因为在生成初始参数时，随机变量(normal distribution)会比全部为0要好很多，
    # 所以我们这里的weights为一个in_size行, out_size列的随机变量矩阵。
    # 注解1: 这里是矩阵乘法，我们输入的是一个in_size大小的向量，要输出一个out_size的向量当然要这么乘
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))

    # 在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    # 神经网络未激活的值
    # tf.matmul()是矩阵的乘法。
    # 即 y = x*W + b
    # 注: 这里的x不需要转置
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    # 没有额外的激励函数的话，返回值就是上面等式的结果
    # 如果有激励函数，结果会输入到激励函数得到结果
    # 即 y = AF(x*W + b)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    
    return outputs


# ----------------------------------------------------------------------------------------
# 构建所需的数据。 
# 这里的x_data和y_data并不是严格的一元二次函数的关系，
# 因为我们多加了一个noise, 这样看起来会更像真实情况。
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# 我们生成的数据是300行1列
print_nparray("x_data", x_data, 5)
print_nparray("noise", noise, 5)
print_nparray("y_data", y_data, 5)

# [None, 1]表示的是数据的shape
# 多少行都无所谓，但是只有一维
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


# ----------------------------------------------------------------------------------------
# 定义神经层
# 假设隐藏层有10个神经元
# 因为只有一个输入和一个输出
# 所以结构是 输入层1个、隐藏层10个、输出层1个

# 定义隐藏层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 定义输出层
# 输入是隐藏层的输出 l1
# 输入有10层，输出有1层（这个解释不科学啊）
prediction = add_layer(l1, 10, 1, activation_function=None)


# ----------------------------------------------------------------------------------------
# 定义loss function
# 但是后面的参数我不知道啥意思啊
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

# 优化器(如果要加快训练速度和提高效果的话要从这个地方下功夫)
# 让机器学习提升它的准确率
# 0.1指的是当前的学习效率（最小化loss）
# 以及这个学习率是如何影响最终结果的（过拟合和欠拟合）?
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化
init = tf.global_variables_initializer() 
sess = tf.Session()
sess.run(init)


# ----------------------------------------------------------------------------------------
# 训练

# 所以训练这里到底发生了什么
# 是每一次都会走这11层么？
# 为啥没有输入层?
# 是通过结果来修改内部存储的W和b?
# 所以是只有第一次的时候是随机的W和b?
for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))