import numpy as np
import os
import random
import warnings
warnings.filterwarnings("ignore")
from model import *
from dataloader import *
from train import *
from plot import *

if __name__ == '__main__':
    train_x, train_y, val_x, val_y, test_x, test_y = load_MNIST('MNIST/raw')
    bx, by = get_batch_data(train_x, train_y, 32)
    vx, vy = get_batch_data(val_x, val_y, 32)
    tx, ty = get_batch_data(test_x, test_y, 32)

    ###final model
    finalmodel = NeuralNetwork(784, 128, 10, 0.5, 0.05, 0.001)
    loss_record, acc_record = train(finalmodel, bx, by, tx, ty, 30, 256, printf=True, record=True)

    PlotResult(loss_record, acc_record)
    PlotVisualization(finalmodel.w1, 'w1')
    PlotVisualization(finalmodel.w2, 'w2')

