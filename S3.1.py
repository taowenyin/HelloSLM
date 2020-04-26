import numpy as np


class KNNModel:
    """
    input_x：实例数据集
    input_y：标签结果
    k：近邻数
    S：数据点
    """
    def __init__(self, input_x, input_y, k, s):
        self._input_x = input_x
        self._input_y = input_y
        self._s = s

    def train(self):
        # 以map形式保存距离
        d = {}
        for i in range(len(self._input_x)):
            # 计算欧氏距离
            l = np.sqrt(np.sum(self._input_x[i] - self._s)**2)
            # 保存距离
            d[l] = i

        # 升序排序后的结果
        d_sorted = sorted(d.keys())
        # 得到最近的那个索引
        index = d[d_sorted[0]]

        # 打印分类
        print(self._input_y[index])


if __name__ == '__main__':
    train_x = np.array([
        [5, 4],
        [9, 6],
        [4, 7],
        [2, 3],
        [8, 1],
        [7, 2],
    ])
    train_y = np.array([1, 1, 1, -1, -1, -1])
    s = np.array([(5, 3)])

    knn = KNNModel(train_x, train_y, 1, s)
    knn.train()