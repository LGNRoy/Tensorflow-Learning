import tensorflow as tf

# create two matrixes
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
# 矩阵的乘积
product = tf.matmul(matrix1,matrix2) 

# 用sessionn来激活product
# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()
# [[12]]

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
# [[12]]