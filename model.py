import numpy as np
import os
import random
import warnings
warnings.filterwarnings("ignore")

###二层神经网络分类器
class NeuralNetwork(object):

    def __init__(self, input_channels, hidden_channels, output_channels, lambda1, lambda2, learning_rate):
        super(NeuralNetwork, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.w1 = np.random.randn(self.input_channels, self.hidden_channels)
        self.w2 = np.random.randn(self.hidden_channels, self.output_channels)
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return x * (1 - x)

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

    def lr_decay(self, epoch):
        return 0.95 ** epoch

    def forward(self, x):
        x = x.reshape(-1, self.input_channels)  # [batch,input_channels]
        self.a1 = np.matmul(x, self.w1)  # [batch,hidden_channels]
        self.z1 = self.sigmoid(self.a1)
        self.a2 = np.matmul(self.z1, self.w2)  # [batch,output_channels]
        self.z2 = self.sigmoid(self.a2)
        output = self.softmax(self.z2)
        return output

    def backward(self, x, y, output, epoch):
        # num_classes == output_channels
        y_onehot = np.eye(self.output_channels)[y].reshape(-1, self.output_channels)  # [batch,num_classes]
        dL_dz2 = output - y_onehot  # [batch,num_classes]
        dz2_da2 = self.dsigmoid(self.z2)  # [batch,num_classes]
        da2_dw2 = self.z1  # [batch,hidden_channels]
        da2_dz1 = self.w2  # [input_channels,hidden_channels]
        dz1_da1 = self.dsigmoid(self.z1)  # [batch,hiden_channels]
        da1_dw1 = x  # [batch,input_channels]

        # renew the parameters
        self.w2 -= (da2_dw2.T.dot(dL_dz2 * dz2_da2) + self.lambda2 * self.w2) * self.learning_rate * 0.9 ** (
                    epoch // 10)
        self.w1 -= (da1_dw1.T.dot(dL_dz2 * dz2_da2).dot(
            da2_dz1.T) + self.lambda1 * self.w1) * self.learning_rate * 0.9 ** (epoch // 10)

    def evaluate(self, x, y):
        output = self.forward(x)
        y_pred = np.argmax(output, axis=1)
        y = y.reshape(-1)
        acc_num = (y_pred == y).sum()

        loss = 0.0
        for i in range(y.shape[0]):
            loss -= np.log(output[i][y[i]])

        loss += 0.5 * self.lambda1 * np.sum(self.w1.T.dot(self.w1))
        loss += 0.5 * self.lambda2 * np.sum(self.w2.T.dot(self.w2))
        return loss / y.shape[0], acc_num

