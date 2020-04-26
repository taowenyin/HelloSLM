import numpy as np
import time

from sklearn.svm import SVC

if __name__ == '__main__':
    # 计算开始时间
    star = time.time()

    data = np.array([
        [1, 2, 1], [2, 3, 1], [3, 3, 1],
        [2, 1, -1], [3, 2, -1]
    ])

    train_x = data[:, 0:2]
    train_y = data[:, 2]

    clf = SVC(C=100, kernel='linear')
    clf.fit(train_x, train_y)

    # 获取w
    w = clf.coef_[0]
    # 获取截距
    b = clf.intercept_

    # 打印超平面
    print(clf.support_vectors_)
    print(w, b)

    # 计算结束事时间
    end = time.time()
    print('用时：{:.3f}s'.format(end - star))