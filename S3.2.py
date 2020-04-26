import numpy as np

from sklearn.neighbors import KNeighborsClassifier

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

    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    knn.fit(train_y, train_y)
    y = knn.predict(s)

    print('Predict Y = %s' % y)
