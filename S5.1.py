import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    train_data = pd.DataFrame([
        [1, '青年', '否', '否', '一般', '否'],
        [2, '青年', '否', '否', '好', '否'],
        [3, '青年', '是', '否', '好', '是'],
        [4, '青年', '是', '是', '一般', '是'],
        [5, '青年', '否', '否', '一般', '否'],
        [6, '中年', '否', '否', '一般', '否'],
        [7, '中年', '否', '否', '好', '否'],
        [8, '中年', '是', '是', '好', '是'],
        [9, '中年', '否', '是', '非常好', '是'],
        [10, '中年', '否', '是', '非常好', '是'],
        [11, '老年', '否', '是', '非常好', '是'],
        [12, '老年', '否', '是', '好', '是'],
        [13, '老年', '是', '否', '好', '是'],
        [14, '老年', '是', '否', '非常好', '是'],
        [15, '老年', '否', '否', '一般', '否'],
    ])

    test_data = pd.DataFrame([
        [1, '青年', '否', '是', '一般'],
        [2, '中年', '是', '否', '好'],
        [3, '老年', '否', '是', '一般'],
    ])

    # 获取训练数据的X和Y
    x_train = train_data.iloc[:, 1:-1]
    y_train = train_data.iloc[:, -1]
    x_test = test_data.iloc[:, 1:test_data.shape[1]]

    # 获取数据中的数据类别
    x_unique = np.unique(x_train)
    y_unique = np.unique(y_train)

    # 标签编码预处理
    x_encoder = LabelEncoder()
    y_encoder = LabelEncoder()

    # 拟合X和Y的标签
    x_encoder.fit(x_unique)
    y_encoder.fit(y_unique)

    # 把Y转化为标签编码
    y_train = y_encoder.transform(y_train)
    # 把X转化为标签编码
    for i in range(x_train.shape[1]):
        x = x_train.iloc[:, i].values
        value = x_encoder.transform(x_train.iloc[:, i])
        data = dict(map(lambda x, y: [x, y], x, value))
        x_train.iloc[:, i].replace(data, inplace=True)
    x_train = x_train.values

    # 把X转化为标签编码
    for i in range(x_test.shape[1]):
        x = x_test.iloc[:, i].values
        value = x_encoder.transform(x_test.iloc[:, i])
        data = dict(map(lambda x, y: [x, y], x, value))
        x_test.iloc[:, i].replace(data, inplace=True)
    x_test = x_test.values

    # 创建分类器
    clf = DecisionTreeClassifier()
    # 训练数据拟合
    clf.fit(x_train, y_train)
    # 对测试数据进行预测
    y_test = clf.predict(x_test)
    # 反向获取描述
    y_test = y_encoder.inverse_transform(y_test)

    print(y_test)

