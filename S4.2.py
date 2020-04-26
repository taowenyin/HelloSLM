import numpy as np

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

if __name__ == '__main__':
    data = np.array([
        [1, 'S', -1], [1, 'M', -1], [1, 'M', 1], [1, 'S', 1], [1, 'S', -1],
        [2, 'S', -1], [2, 'M', -1], [2, 'M', 1], [2, 'L', 1], [2, 'L', 1],
        [3, 'L', 1], [3, 'M', 1], [3, 'M', 1], [3, 'L', 1], [3, 'L', -1],
    ])

    # 数据
    train_x = data[:, [0, 1]]
    # 标签
    train_y = data[:, 2].astype(np.int)

    # 批量替换
    train_x[train_x == 'S'] = 4
    train_x[train_x == 'M'] = 5
    train_x[train_x == 'L'] = 6
    # 类型转换
    train_x = train_x.astype(np.int)

    # 测试数据
    x = np.array([[2, 'S']])
    x[x == 'S'] = 4
    x = x.astype(np.int)

    # 高斯朴素贝叶斯对象
    gaussianNB = GaussianNB()
    gaussianNB.fit(train_x, train_y)
    print('Gaussian Test X = ', gaussianNB.predict(x))
    print('Gaussian Test X = ', gaussianNB.predict_proba(x))

    # 伯努利朴素贝叶斯
    bernoulliNB = BernoulliNB()
    bernoulliNB.fit(train_x, train_y)
    print('Bernoulli Test X = ', bernoulliNB.predict(x))
    print('Bernoulli Test X = ', bernoulliNB.predict_proba(x))

    # 多项式朴素贝叶斯
    multinomialNB = MultinomialNB()
    multinomialNB.fit(train_x, train_y)
    print('Multinomial Test X = ', multinomialNB.predict(x))
    print('Multinomial Test X = ', multinomialNB.predict_proba(x))

