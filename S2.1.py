import numpy as np


# 感知机类
class PerceptronModel:
    """
    input_x：实例数据集
    input_y：标签结果
    learn_rate：学习率
    """
    def __init__(self, input_x, input_y, learn_rate=1.):
        self._input_x = input_x
        self._input_y = input_y
        self._learn_rate = learn_rate
        # 感知机的权值
        self._w = 0
        # 感知机的偏置
        self._b = 0

    # 原始形式算法
    def sgd_train(self):
        feature_number = self._input_x[0].shape[0]
        # 根据数据的特征数初始化权值
        self._w = np.zeros(feature_number)

        while True:
            finish = True
            for i in range(len(self._input_x)):
                # 求权值和输入
                inner = np.inner(self._w, self._input_x[i])
                if (self._input_y[i] * (inner + self._b)) <= 0:
                    # 说明不是所有值都被线性分割
                    finish = False
                    # 更新参数
                    self._w += (self._learn_rate * self._input_y[i] * self._input_x[i])
                    self._b += (self._learn_rate * self._input_y[i])

            if finish:
                # 全部都已经被线性分割
                break
            else:
                # 继续计算
                continue

        print('Final W = %s, B = %s' % (self._w, self._b))


if __name__ == '__main__':
    input_x = np.array([
        [3, 3],
        [4, 3],
        [1, 1],
    ])
    input_y = np.array([1, 1, -1])

    # 创建模型对象
    model = PerceptronModel(input_x, input_y)
    # 执行原始形式训练
    model.sgd_train()