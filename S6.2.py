import numpy as np
import pandas as pd
import time

from sklearn.linear_model import LogisticRegression


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

    logisticRegression = LogisticRegression(max_iter=1000)
    # 拟合数据
    logisticRegression.fit(train_x, train_y)
    # 预测数据
    print('训练集的准确率：{}'.format(logisticRegression.score(train_x, train_y)))
    print('预测结果：{}'.format(logisticRegression.predict(test_x)))

    # 计算结束事时间
    end = time.time()
    print('用时：{:.3f}s'.format(end - star))