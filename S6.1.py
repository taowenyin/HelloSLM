import numpy as np
import pandas as pd
import time

# 逻辑斯谛回归模型
class LogisticRegression:
    """
    input_x：输入训练特征
    input_y：训练结果
    learn_rate：学习率
    max_iter：最大迭代次数
    tol：迭代停止阈值
    """
    def __init__(self, input_x, input_y, learn_rate=0.1, max_iter=10000, tol=1e-2):
        self.input_x = input_x
        self.input_y = input_y.reshape(len(input_y), 1)
        self.learn_rate = learn_rate
        self.max_iter = max_iter
        self.tol = tol
        self.w = 0

        # 对数据进行预处理
        self.input_x = self.preprocessing(input_x)

    # 数据预处理
    def preprocessing(self, x):
        # 获取特征数量
        row = x.shape[0]
        # 获取X的最后一列数据
        one_data = np.ones(row).reshape(row, 1)
        # 把wx+b->w
        x = np.hstack((x, one_data))

        return x

    # 计算导后方程中的sigmod
    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self):
        # 初始化权重
        self.w = np.zeros(self.input_x.shape[1], dtype=np.float).reshape(1, self.input_x.shape[1])
        # 记录迭代次数
        times = 0

        for i in range(self.max_iter):
            # 计算梯度值
            z = self.sigmod(np.dot(self.input_x, self.w.T))
            grad = self.input_x * (self.input_y - z)
            # 求每个分量的梯度和
            grad = grad.sum(axis=0)
            if (np.abs(grad) <= self.tol).all():
                break
            else:
                self.w += self.learn_rate * grad
                times += 1

        print('迭代次数：{}次'.format(times))
        print('最终权重：{}次'.format(self.w))

    # 预测实例
    def predict(self, x):
        # 计算预测的概率
        p = self.sigmod(np.dot(self.preprocessing(x), self.w.T))
        print('Y = 1的概率为：{:.2%}'.format(p[0, 0]))

        # 计算结果
        if p[0, 0] > 0.5:
            p = 1
        else:
            p = 0

        return p


if __name__ == '__main__':
    # 计算开始时间
    star = time.time()

    data = np.array([
        [3, 3, 3, 1], [4, 3, 2, 1], [2, 1, 2, 1],
        [1, 1, 1, 0], [-1, 0, 1, 0], [2, -2, 1, 0],
    ])

    test_x = np.array([[1, 2, -2]])

    train_x = data[:, 0:3]
    train_y = data[:, 3]

    logisticRegression = LogisticRegression(train_x, train_y)
    # 拟合数据
    logisticRegression.fit()
    # 预测数据
    print(logisticRegression.predict(test_x))

    # 计算结束事时间
    end = time.time()
    print('用时：{:.3f}s'.format(end - star))