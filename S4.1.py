import numpy as np
import pandas as pd


if __name__ == '__main__':
    data = np.array([
        [1, 'S', -1], [1, 'M', -1], [1, 'M', 1], [1, 'S', 1], [1, 'S', -1],
        [2, 'S', -1], [2, 'M', -1], [2, 'M', 1], [2, 'L', 1], [2, 'L', 1],
        [3, 'L', 1], [3, 'M', 1], [3, 'M', 1], [3, 'L', 1], [3, 'L', -1],
    ])

    '''
    贝叶斯估计实现
    Begin
    '''
    # 平滑系数
    lamd = 0.2

    # 把Y值转化为DF格式
    y_df = pd.DataFrame(data[:, 2])
    # Y值的统计信息
    y_value_count = pd.value_counts(y_df.iloc[:, 0])
    # 计算Y的类别数量
    category = y_value_count.index.size

    # 把X值转化为DF格式
    x_df = pd.DataFrame(data[:, 0:data.shape[1] - 1])
    # 保存X值的统计信息
    x_value_count = []
    for i in range(x_df.shape[1]):
        item_count = pd.DataFrame(pd.value_counts(x_df.iloc[:, i]))
        x_value_count.append(item_count)

    # 保存每列的情况
    y_column_probability = {}
    # 计算Y的种类
    for i in range(y_value_count.index.size):
        # 当前Y的标签
        y_key = y_value_count.index[i]
        # 当前Y的数量
        y_count = y_value_count.iloc[i]

        # 获取Y=y_key的数据
        x_data = pd.DataFrame(data[data[:, data.shape[1] - 1] == y_key][:, [0, data.shape[1] - 2]])

        # 检查每一列的数据
        x_column_item = {}
        for j in range(x_data.shape[1]):
            x_column = pd.DataFrame(x_data.iloc[:, j])
            # 每列数据的种类
            x_column_category = x_value_count[j]

            # 计算每列数据中，每种数据的数量
            x_column_sub_item = {}
            for k in range(x_column_category.index.size):
                # 数据的种类
                key = x_column_category.index[k]
                # 对应的数量
                size = x_column[x_column.iloc[:, 0] == key].size
                # x_column_sub_item[key] = size
                x_column_sub_item[key] = (size + lamd) / (y_count + x_column_category.size * lamd)

            # 保存每个x的值
            x_column_item[j] = x_column_sub_item
        # 保存每种Y的值
        y_column_probability[y_key] = x_column_item

    # 保存概率字典
    y_probability = {}
    # 计算先验概率的贝叶斯估计
    for i in range(y_value_count.index.size):
        y_probability[y_value_count.index[i]] = (y_value_count.iloc[i] + lamd) / ( y_df.size + y_value_count.index.size * lamd)
    '''
    贝叶斯估计实现
    End
    '''

    x = np.array([2, 'S'])
    p_positive = y_probability['1'] * y_column_probability['1'][0][str(x[0])] * y_column_probability['1'][1][x[1]]
    p_negative = y_probability['-1'] * y_column_probability['-1'][0][str(x[0])] * y_column_probability['-1'][1][x[1]]

    if p_positive > p_negative:
        print('x is positive')
    else:
        print('x is negative')