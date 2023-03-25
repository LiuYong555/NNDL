import numpy as np
import os
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import warnings
warnings.filterwarnings("ignore")

def PlotVisualization(w,title):
    x = np.arange(w.shape[0])
    y = np.arange(w.shape[1])
    X,Y = np.meshgrid(x,y)
    Z = w.T
    plt.figure(figsize=(6,6))
    ax = plt.axes(projection='3d')
    ax.view_init(45, 35)
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('result/visualization{}.png'.format(title))


def PlotAcc(**records):
    plt.figure(figsize=(8, 6))
    for record in records.items():
        label, data = record[0], record[1]
        xticks = ['32', '64', '128', '256', '512', '1024']
        plt.plot(xticks, data, marker='o', label=label)
        plt.xlabel("Hidden Layer")
        plt.ylabel("Accuracy")
        plt.legend()
    plt.savefig('result/accuracy.png')


def PlotLoss(**records):
    plt.figure(figsize=(8, 6))
    for record in records.items():
        label, data = record[0], record[1]
        xticks = ['32', '64', '128', '256', '512', '1024']
        plt.plot(xticks, data, marker='o', label=label)
        plt.xlabel("Hidden Layer")
        plt.ylabel("Loss")
        plt.legend()
    plt.savefig('result/loss.png')

def PlotResult(loss_record,acc_record):
    fig=plt.figure(figsize=(8,6))
    ax1=fig.add_subplot(111)
    ax2=ax1.twinx()
    epoch_label=[1]
    for i in range(1,21):
        epoch_label.append(50*i)


    loss_plot = ax1.plot(loss_record,c = 'blue',linewidth = 1)
    acc_plot = ax2.plot(acc_record,c = 'red',linewidth = 1)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Acc')

    lines=loss_plot+acc_plot
    labels=['Loss','Acc']
    plt.title('Loss&Correct in TestingSet')
    plt.legend(lines,labels)
    #plt.grid()
    fig.show()
    plt.savefig('result/testing_set_loss_acc.png')