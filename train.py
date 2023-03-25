import numpy as np
import os
import random
import warnings
warnings.filterwarnings("ignore")
from model import NeuralNetwork
from dataloader import *


def train(model, train_x, train_y, val_x, val_y, epochs, mini_rate, printf=False, record=False):
    # 记录验证集上的loss与acc
    val_loss_record = []
    val_acc_record = []
    for epoch in range(epochs):
        minibatch_x, minibatch_y = get_minibatch(train_x, train_y, mini_rate)

        for x, y in zip(minibatch_x, minibatch_y):
            y = np.array(y.reshape(-1, 1), dtype=np.dtype(np.int32))

            output = model.forward(x)
            model.backward(x, y, output, epoch)

        total_train_loss = 0.0
        total_train_acc = 0
        for x, y in zip(train_x, train_y):
            y = np.array(y.reshape(-1, 1), dtype=np.dtype(np.int32))
            loss, acc_num = model.evaluate(x, y)
            total_train_loss += loss
            total_train_acc += acc_num

        total_val_loss = 0.0
        total_val_acc = 0.0
        for x, y in zip(val_x, val_y):
            y = np.array(y.reshape(-1, 1), dtype=np.dtype(np.int32))
            loss, acc_num = model.evaluate(x, y)
            total_val_loss += loss
            total_val_acc += acc_num

        val_loss_record.append(total_val_loss)
        val_acc_record.append(total_val_acc)
        if printf == True:
            print('Epoch:{}/{},Train loss:{} Train acc:{} Val loss:{} Val acc:{}'.format(
                epoch + 1, epochs, round(total_train_loss / 50000, 3), round(total_train_acc / 50000, 3),
                round(total_val_loss / 10000, 3), round(total_val_acc / 10000, 3)))
    if record == True:
        return [i / 10000 for i in val_loss_record], [i / 10000 for i in val_acc_record]
    else:
        return val_loss_record[-1] / 10000, val_acc_record[-1] / 10000