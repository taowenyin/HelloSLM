import numpy as np
import time

from sklearn.ensemble import AdaBoostClassifier

# AdaBoost
if __name__ == '__main__':
    # 计算开始时间
    star = time.time()

    data = np.array([
        [0, 1], [1, 1], [2, 1], [3, -1], [4, -1], [5, -1], [6, 1], [7, 1], [8, 1], [9, -1]
    ])

    train_x = data[:, 0].reshape(10, 1)
    train_y = data[:, 1]

    clf = AdaBoostClassifier()
    clf.fit(train_x, train_y)
    y_predict = clf.predict(train_x)
    score = clf.score(train_x, train_y)

    # 计算结束事时间
    end = time.time()
    print('用时：{:.3f}s'.format(end - star))