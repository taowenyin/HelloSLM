import numpy as np
import time


# AdaBoost算法类
class AdaBoost:
    def __init__(self, x, y, tol=0.05, max_iter=10):
        self.x = x
        self.y = y
        self.tol = tol
        self.max_iter = max_iter
        # 初始化样本权重：（1）步骤
        self.w = np.full(x.shape[0], 1 / x.shape[0])
        # 弱分类器的权重
        self.alpha = []
        # 弱分类器
        self.G = []
        # 分类器阈值的上下限
        self.v_min = min(x) - 0.5
        self.v_max = max(x) + 0.5

        self._select_v()

    # 计算阈值，返回阈值，符号，和误分类值
    def _select_v(self):
        # 误分类的最小值
        e_min = np.inf
        # 小于阈值的样本分类
        sign = None
        # 最佳的阈值
        v_best = None

        # 遍历所有可能的阈值
        for v in np.arange(self.v_min, self.v_max + 0.5, 1):
            # 计算小于和大于阈值，把错分类的数据和样本权重相乘：（3）（b）步骤
            e_1 = abs((self.y[self.x < v] - 1) * self.w[self.x < v]) # 由于采用了+1、-1方式，因此最终值为原值的2倍，并且有正负号，需要取绝对值
            e_2 = abs((self.y[self.x > v] + 1) * self.w[self.x > v])
            e = (e_1.sum() + e_2.sum()) / 2 # 计算误分类值
            # 若误差小于0.5，那么说明x < v时，y = 1的假设是正确的，否则就是错误的
            if e < 0.5:
                flag = 1
            else:
                e = 1 - e
                flag = -1

            # 寻找最小的误分类，并保存误分类值、阈值和分类
            if e < e_min:
                e_min = e
                sign = flag
                v_best = v

        return v_best, sign, e_min

    def fit(self):
        G = 0
        for i in range(self.max_iter):
            # 获得最佳参数
            v_best, sign, e_min = self._select_v()
            # 计算基础模型的alpha系数：（2）（c）步骤
            alpha = 1 / 2 * np.log((1 - e_min) / e_min)
            # 保存模型alpha系数
            self.alpha.append(alpha)
            # 保存模型弱分类器参数
            self.G.append((v_best, sign))
            # 计算基本分类器的结果：（3）步骤
            _G = self.base_estimator(self.x, i)
            # 把基本分类器的结果与对应的模型权重相乘
            G = G + alpha * _G
            # 计算总模型的预测结果
            y_predict = np.sign(G)
            # 计算误差率
            error_rate = np.sum(np.abs(y_predict - self.y)) / 2 / self.y.shape[0]
            # 如果误差率小于设定值，则结束
            if error_rate < self.tol:
                print('迭代次数:', i + 1)
                break
            else:
                # 更新权重
                self._update_w()


    # 选择第i分类器进行预测
    def base_estimator(self, x, i):
        # 获取基本模型的数据
        v_best, sign = self.G[i]
        _G = x.copy()
        # 对输入数据进行预测，并返回预测值
        _G[_G < v_best] = sign
        _G[_G > v_best] = -sign
        return _G

    # 更新权重
    def _update_w(self):
        # 获取最后一个模型的信息
        v_best, sign = self.G[-1]
        # 获取最后一个模型的alpha
        alpha = self.alpha[-1]
        # 重建分类器结果
        G = np.zeros(self.y.size, dtype=int)
        G[self.x < v_best] = sign
        G[self.x > v_best] = -sign

        # 计算更新的权重：（2）（d）步骤
        _w = self.w * np.exp(-1 * alpha * self.y * G)
        self.w = _w / _w.sum()

    # 对数据进行预测
    def predict(self, x):
        G = 0
        for i in range(len(self.alpha)):
            _G = self.base_estimator(x, i)
            alpha = self.alpha[i]
            G = G + alpha * _G

        y_predict = np.sign(G)
        return y_predict.astype(int)

    # 计算分数
    def score(self, x, y):
        y_predict = self.predict(x)
        error_rate = np.sum(np.abs(y_predict - self.y)) / 2 / self.y.shape[0]
        return 1 - error_rate

# AdaBoost
if __name__ == '__main__':
    # 计算开始时间
    star = time.time()

    data = np.array([
        [0, 1], [1, 1], [2, 1], [3, -1], [4, -1], [5, -1], [6, 1], [7, 1], [8, 1], [9, -1]
    ])

    train_x = data[:, 0]
    train_y = data[:, 1]

    clf = AdaBoost(train_x, train_y)
    clf.fit()
    y_predict = clf.predict(train_x)
    score = clf.score(train_x, train_y)

    # 计算结束事时间
    end = time.time()
    print('用时：{:.3f}s'.format(end - star))