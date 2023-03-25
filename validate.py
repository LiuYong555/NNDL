import numpy as np
import os
import random
import warnings
warnings.filterwarnings("ignore")
from model import NeuralNetwork
from dataloader import *
from train import *
from plot import *

def RegularizationSelect(lambda_range):
    acc_matrix = np.zeros((len(lambda_range), len(lambda_range)))
    loss_matrix = np.zeros((len(lambda_range), len(lambda_range)))
    for i in range(len(lambda_range)):
        for j in range(len(lambda_range)):
            loss, acc = train(NeuralNetwork(784, 128, 10, lambda_range[i], lambda_range[j], 0.001), bx, by, vx, vy, 30, 256)
            loss_matrix[i][j] = loss
            acc_matrix[i][j] = acc
    return loss_matrix, acc_matrix

if __name__ == '__main__':
    train_x, train_y, val_x, val_y, test_x, test_y = load_MNIST('MNIST/raw')

    bx, by = get_batch_data(train_x, train_y, 32)
    vx, vy = get_batch_data(val_x, val_y, 32)
    tx, ty = get_batch_data(test_x, test_y, 32)
    # 学习率固定为0.001时，用不同隐藏层数训练模型
    model_h32_lr001 = NeuralNetwork(784, 32, 10, 0.5, 0.5, 0.001)
    h32_lr001_loss, h32_lr001_acc = train(model_h32_lr001, bx, by, vx, vy, 30, 256)

    model_h64_lr001 = NeuralNetwork(784, 64, 10, 0.5, 0.5, 0.001)
    h64_lr001_loss, h64_lr001_acc = train(model_h64_lr001, bx, by, vx, vy, 30, 256)

    model_h128_lr001 = NeuralNetwork(784, 128, 10, 0.5, 0.5, 0.001)
    h128_lr001_loss, h128_lr001_acc = train(model_h128_lr001, bx, by, vx, vy, 30, 256)

    model_h256_lr001 = NeuralNetwork(784, 256, 10, 0.5, 0.5, 0.001)
    h256_lr001_loss, h256_lr001_acc = train(model_h256_lr001, bx, by, vx, vy, 30, 256)

    model_h512_lr001 = NeuralNetwork(784, 512, 10, 0.5, 0.5, 0.001)
    h512_lr001_loss, h512_lr001_acc = train(model_h512_lr001, bx, by, vx, vy, 30, 256)

    model_h1024_lr001 = NeuralNetwork(784, 1024, 10, 0.5, 0.5, 0.001)
    h1024_lr001_loss, h1024_lr001_acc = train(model_h1024_lr001, bx, by, vx, vy, 30, 256)

    # 学习率固定为0.005时，用不同隐藏层数训练模型
    model_h32_lr005 = NeuralNetwork(784, 32, 10, 0.5, 0.5, 0.005)
    h32_lr005_loss, h32_lr005_acc = train(model_h32_lr005, bx, by, vx, vy, 20, 256)

    model_h64_lr005 = NeuralNetwork(784, 64, 10, 0.5, 0.5, 0.005)
    h64_lr005_loss, h64_lr005_acc = train(model_h64_lr005, bx, by, vx, vy, 20, 256)

    model_h128_lr005 = NeuralNetwork(784, 128, 10, 0.5, 0.5, 0.005)
    h128_lr005_loss, h128_lr005_acc = train(model_h128_lr005, bx, by, vx, vy, 20, 256)

    model_h256_lr005 = NeuralNetwork(784, 256, 10, 0.5, 0.5, 0.005)
    h256_lr005_loss, h256_lr005_acc = train(model_h256_lr005, bx, by, vx, vy, 20, 256)

    model_h512_lr005 = NeuralNetwork(784, 512, 10, 0.5, 0.5, 0.005)
    h512_lr005_loss, h512_lr005_acc = train(model_h512_lr005, bx, by, vx, vy, 20, 256)

    model_h1024_lr005 = NeuralNetwork(784, 1024, 10, 0.5, 0.5, 0.005)
    h1024_lr005_loss, h1024_lr005_acc = train(model_h1024_lr005, bx, by, vx, vy, 20, 256)

    # 学习率固定为0.01时，用不同隐藏层数训练模型
    model_h32_lr01 = NeuralNetwork(784, 32, 10, 0.5, 0.5, 0.01)
    h32_lr01_loss, h32_lr01_acc = train(model_h32_lr01, bx, by, vx, vy, 20, 256)

    model_h64_lr01 = NeuralNetwork(784, 64, 10, 0.5, 0.5, 0.01)
    h64_lr01_loss, h64_lr01_acc = train(model_h64_lr01, bx, by, vx, vy, 20, 256)

    model_h128_lr01 = NeuralNetwork(784, 128, 10, 0.5, 0.5, 0.01)
    h128_lr01_loss, h128_lr01_acc = train(model_h128_lr01, bx, by, vx, vy, 20, 256)

    model_h256_lr01 = NeuralNetwork(784, 256, 10, 0.5, 0.5, 0.01)
    h256_lr01_loss, h256_lr01_acc = train(model_h256_lr01, bx, by, vx, vy, 20, 256)

    model_h512_lr01 = NeuralNetwork(784, 512, 10, 0.5, 0.5, 0.01)
    h512_lr01_loss, h512_lr01_acc = train(model_h512_lr01, bx, by, vx, vy, 20, 256)

    model_h1024_lr01 = NeuralNetwork(784, 1024, 10, 0.5, 0.5, 0.01)
    h1024_lr01_loss, h1024_lr01_acc = train(model_h1024_lr01, bx, by, vx, vy, 20, 256)

    #绘制结果
    lr001_acc = [h32_lr001_acc, h64_lr001_acc, h128_lr001_acc, h256_lr001_acc, h512_lr001_acc, h1024_lr001_acc]
    lr005_acc = [h32_lr005_acc, h64_lr005_acc, h128_lr005_acc, h256_lr005_acc, h512_lr005_acc, h1024_lr005_acc]
    lr01_acc = [h32_lr01_acc, h64_lr01_acc, h128_lr01_acc, h256_lr01_acc, h512_lr01_acc, h1024_lr01_acc]
    acc_dic = {'lr=0.001': lr001_acc, 'lr=0.005': lr005_acc, 'lr=0.01': lr01_acc}
    PlotAcc(**acc_dic)

    lr001_loss = [h32_lr001_loss, h64_lr001_loss, h128_lr001_loss, h256_lr001_loss, h512_lr001_loss, h1024_lr001_loss]
    lr005_loss = [h32_lr005_loss, h64_lr005_loss, h128_lr005_loss, h256_lr005_loss, h512_lr005_loss, h1024_lr005_loss]
    lr01_loss = [h32_lr01_loss, h64_lr01_loss, h128_lr01_loss, h256_lr01_loss, h512_lr01_loss, h1024_lr01_loss]
    loss_dic = {'lr=0.001': lr001_loss, 'lr=0.005': lr005_loss, 'lr=0.01': lr01_loss}
    PlotLoss(**loss_dic)

    lambda_range = [0.01, 0.05, 0.1, 0.5, 1]
    loss_matrix, acc_matrix = RegularizationSelect(lambda_range)
    print(loss_matrix)
    print(acc_matrix)

