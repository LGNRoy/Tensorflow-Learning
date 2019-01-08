import tensorflow as tf

#在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# mul = multiply 是将input1和input2 做乘法运算，并输出为 output 
ouput = tf.multiply(input1, input2)

# 接下来, 传值的工作交给了 sess.run(), 
# 需要传入的值放在了feed_dict={} 
# 并一一对应每一个 input. 
# placeholder 与 feed_dict={} 是绑定在一起出现的。
with tf.Session() as sess:
    print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))
# [ 14.]