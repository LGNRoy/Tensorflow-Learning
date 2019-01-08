import tensorflow as tf

# 在tf里面，只有用tf声明了是变量的才是
# 加减乘除也是
# 感觉就是先占个坑

state = tf.Variable(0, name='counter') # 不知道这个name是干什么的

# 定义常量 one
one = tf.constant(1)

# 定义加法步骤 (注: 此步并没有直接计算)
new_value = tf.add(state, one)

# 将 State 更新成 new_value
update = tf.assign(state, new_value)



# 在tf里面设置了变量的话要初始化变量
# 并在session中激活
init = tf.global_variables_initializer() 
 
# 使用 Session
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))