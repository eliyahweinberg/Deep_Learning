import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression


def sigmoid(wx):
    return 1 / (1 + np.exp(-wx))


def log_likelihod(_x, _y, w):
    wx = np.dot(_x, w)
    return np.sum(_y * wx - np.log(1 + np.exp(wx)))


def logistic_regression(_x, _y, num_steps, learning_rate):
    intercept = np.ones((_x.shape[0], 1))
    _x = np.hstack((intercept, _x))
    w = np.zeros(_x.shape[1])
    for step in range(num_steps):
        wx = np.dot(_x, w)
        est_y = sigmoid(wx)
        err = _y - est_y
        gradient = np.dot(_x.T, err)
        w += learning_rate * gradient
        if step % 10000 == 0:
            print(log_likelihod(_x, _y, w))
    return w


if __name__ == '__main__':
    np.random.seed(12)
    num_observation = 5000

    x1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observation)
    x2 = np.random.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observation)

    x = np.vstack((x1, x2)).astype(np.float32)
    y = np.hstack((np.zeros(num_observation), np.ones(num_observation)))

    weights = logistic_regression(x, y, num_steps=100000, learning_rate=5e-5)

    clf = LogisticRegression()
    clf.fit(x, y)

    print(clf.intercept_, clf.coef_)
    print(weights)
    data_with_intercept = np.hstack((np.ones((x.shape[0], 1)), x))
    final_scores = np.dot(data_with_intercept, weights)
    preds = np.round(sigmoid(final_scores))
    print('Accuracy from scratch: {0}'.format((preds == y).sum().astype(float)/len(preds)))
    print('Accuracy from sk-learn: {0}'.format(clf.score(x, y)))
    plt.figure(figsize=(12, 8))
    plt.scatter(x[:, 0], x[:, 1], c=(preds == y) - 1, alpha=.8, s=50)
    plt.show()
