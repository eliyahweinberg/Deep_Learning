import numpy as np


def sigmoid(wx):
    return 1 / (1 + np.exp(-wx))


def sigmoid_(wx):
    return (1-sigmoid(wx))*sigmoid(wx)


if __name__ == '__main__':
    inputLayerSize, hiddenLayerSize, outputLayerSize = 3, 3, 1

    print(sigmoid(-10) - 0.9999 < 6e-4)
    print(sigmoid(10) - 0.9999 < 6e-4)
    print(sigmoid(0) == 0.5)
    print(sigmoid_(0) == 0.25)

    """# Train the network (Forward + backword)"""

    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    Y = np.array([[0], [1], [1], [0]])
    Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
    Wz = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))

    count = 0
    while True:
        L1 = np.dot(X, Wh)
        print(L1.shape == (4, 3))
        H = sigmoid(L1)     # sigmoid first layer results
        print(H.shape == (4, 3))
        L2 = np.dot(H, Wz)  # second layer
        print(L2.shape == (4, 1))

        Z = sigmoid(L2)     # sigmoid second layer results
        print(Z.shape == (4, 1))
        E = Y - Z   # how much we missed (error)
        print(E.shape == (4, 1))

        # backpropogation step
        dZ = sigmoid_(L2)*E    # gradient Z
        print(dZ.shape == (4, 1))

        dH = sigmoid_(L1)*np.dot(dZ, Wz.T)    # gradient H
        print(dH.shape == (4, 3))

        Wz += np.dot(H.T, dZ)    # update output layer weights
        print(Wz.shape == (3, 1))

        Wh += np.dot(X.T, dH)     # update hidden layer weights
        print(Wh.shape == (3, 3))

        count += 1
        if Z[0] < 0.05 and Z[1] > 0.95 and Z[2] > 0.95 and Z[3] < 0.05:
            break

    print('Training steps={}'.format(count))
    # print(Z[0] < 0.05)  # what have we learnt?
    # print(Z[1] > 0.95)  # what have we learnt?
    # print(Z[2] > 0.95)  # what have we learnt?
    # print(Z[3] < 0.05)  # what have we learnt?
