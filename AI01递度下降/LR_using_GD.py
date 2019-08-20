"""

用梯度下降的优化方法来快速解决线性回归问题

"""
import  numpy as np
import  matplotlib.pyplot as plt
import tensorflow as tf

# 构建数据 (随机100个点)
points_num = 100
vectors = []

# 用numpy 正太随机分布函数 生成100 个点
# 这些点的 (x,y)坐标值 对应一个 线性方程   y = 0.1 *  x+ 0.2
# 权重(Weight) 0.1  偏差是(Bias) 0.2

for i in range(points_num):
    x1 = np.random.normal()
    y1 = 0.1*x1 + 0.2 + np.random.normal(0.0,0.04)
    vectors.append([x1,y1])

# 真实的输入
x_data = [v[0] for v in vectors]
# 真实的值
y_data = [v[1] for v in vectors]


# 图1:显示100个数据点

plt.plot(x_data, y_data, 'r*', label="Original data")
plt.title("LR using Gradient Descent")
plt.legend()
plt.show()

# 构建线性回归模型

# 初始权限
W = tf.Variable(tf.random_uniform([1] , -1.0 , 1.0))
# 初始偏差
b = tf.Variable(tf.zeros([1]))

y = W * x_data +b

print(y)
# 定义loss function
# 对Tensor 的所有维度计算 （y-y_data)^2 / N

loss = tf.reduce_mean(tf.square(y-y_data))

# 用梯度下降 来优化 loss function
# 设置学习率
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 创建一个Tensor会话

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)


# 训练 20 步
for step in range(20):
    sess.run(train)
    print("Step=%d , Loss=%f,[Weight=%f Bias=%f]"% (step ,sess.run(loss),sess.run(W),sess.run(b) ))

# 图2 所有的点。和 恰合的线
plt.plot(x_data, y_data, 'r*', label="Original data")
plt.title("LR using Gradient Descent")
plt.plot(x_data,  sess.run(W) * x_data+sess.run(b), '-', label="Line")
plt.legend()
plt.show()

sess.close()




