import numpy as np

from sklearn.linear_model import Perceptron


if __name__ == '__main__':
    input_x = np.array([
        [3, 3],
        [4, 3],
        [1, 1],
    ])
    input_y = np.array([1, 1, -1])

    # 创建模型对象
    model = Perceptron(eta0=1)
    # 拟合训练对象
    model.fit(input_x, input_y)
    print('Final W = %s, B = %s' % (model.coef_, model.intercept_))

    # 创建模型对象
    model = Perceptron(eta0=0.1)
    # 拟合训练对象
    model.fit(input_x, input_y)
    print('Final W = %s, B = %s' % (model.coef_, model.intercept_))
