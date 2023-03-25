import numpy as np
import os
import random
import warnings
warnings.filterwarnings("ignore")


def load_MNIST(path):
    fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_x = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)
    # 在原有训练集中分出10000条作为验证集
    train_x, val_x = train_x[0:50000], train_x[50000:60000]
    train_x = train_x.reshape(-1, 28 * 28)
    val_x = val_x.reshape(-1, 28 * 28)

    fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_y = loaded[8:].reshape((60000)).astype(np.float)
    # 在原有训练集中分出10000条作为验证集
    train_y, val_y = train_y[0:50000], train_y[50000:60000]

    fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_x = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
    test_x = test_x.reshape(-1, 28 * 28)

    fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_y = loaded[8:].reshape((10000)).astype(np.float)

    return train_x / 255, train_y, val_x / 255, val_y, test_x / 255, test_y


def get_batch_data(data_x, data_y, batch_size):
    index = np.random.permutation(data_x.shape[0])
    data_x = data_x[index]
    data_y = data_y[index]
    batch_x = []
    batch_y = []
    start = 0
    while True:
        if start + batch_size < data_x.shape[0]:
            batch_x.append(data_x[start:start + batch_size])
            batch_y.append(data_y[start:start + batch_size])
            start += batch_size
        else:
            batch_x.append(data_x[start:data_x.shape[0]])
            batch_y.append(data_y[start:data_x.shape[0]])
            break
    return batch_x, batch_y


def get_minibatch(batch_x, batch_y, mini_batch_size):
    # random.seed(1234)

    idx_list = random.sample([i for i in range(len(batch_x))], mini_batch_size)
    minibatch_x = [batch_x[idx] for idx in idx_list]
    minibatch_y = [batch_y[idx] for idx in idx_list]

    return minibatch_x, minibatch_y
