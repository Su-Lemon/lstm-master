import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import random
random.seed(10)
import pandas as pd

from lstm_master.configs import config as cfg


def plotLoss(test_losses, train_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(np.array(range(0, len(test_losses))) /
             float(len(test_losses) - 1) * (len(train_losses) - 1),
             np.log(test_losses), label="Test loss")
    plt.plot(np.log(train_losses), label="Train loss")
    plt.title("Training losses over time (on a logarithmic scale)")
    plt.xlabel('Iteration')
    plt.ylabel('log(Loss)')
    plt.legend(loc='best')
    plt.show()


def plotPred(preout, pastSeq):
    for j in range(cfg.FLAGS.test_size):
        plt.figure(figsize=(12, 3))
        for k in range(cfg.FLAGS.output_dim):
            # expected = Y1[cfg.FLAGS.seq_len - 1:, j, k] * (
            #             self.dataSet.dataMinMax[1] - self.dataSet.dataMinMax[
            #         0]) + self.dataSet.dataMinMax[0]  # 对应预测值的打印
            pred = preout[:, j, k]

            # saveData(pred, k)

            label1 = "Past" if k == 0 else "_nolegend_"
            label2 = "Future" if k == 0 else "_nolegend_"
            label3 = "Pred" if k == 0 else "_nolegend_"
            connect = pastSeq[k]
            connect.append(pred[0])
            plt.plot(range(1854, len(connect)+1854), connect, "-b", label=label1)
            plt.plot(range(len(pastSeq[k])+1854, len(pastSeq[k]) + len(pred)+1854), pred, "-r", label=label3)

        # plt.ylim(-1.8, -1.7)
        plt.xlim(1855,)
        plt.xlabel("Date/year")
        plt.ylabel("SST/degrees")
        plt.legend(loc='best')
        plt.title("SST over the years and prediction")
        plt.show()


def plotSSTMap(pred, all_data):
    X = np.linspace(0, 179, 180)
    Y = np.linspace(0, 88, 89)
    X, Y = np.meshgrid(X, Y)
    Z = pred[-1, 0, :]
    Z = np.reshape(Z, [cfg.FLAGS3[1]-cfg.FLAGS3[0], cfg.FLAGS3[3]-cfg.FLAGS3[2]])
    all_data[-1, cfg.FLAGS3[0]:cfg.FLAGS3[1], cfg.FLAGS3[2]:cfg.FLAGS3[3]] = Z

    plt.matshow(all_data[-1, :, :], cmap=plt.cm.coolwarm, vmin=10, vmax=30)
    plt.colorbar()
    plt.contour(X, Y, all_data[-1, :, :], 15)
    plt.show()


def saveData(list, colInfo):
    df = pd.DataFrame(list)
    df.to_csv("./predSST.csv", index=False, header=None, mode='a')


def loadData():
    label = ["Iceland", "Scotland", "Norway", "Arctic"]
    line = ["-y", "-r", "-g", "-g"]
    for k in range(4):
        data = pd.read_csv("./{}.csv".format(k), names=cfg.FLAGS2['sst'], header=None, encoding="gbk")
        past = data[["last"]].values
        pred = data[["pred"]].values
        label1 = "Past SST" if k == 0 else "_nolegend_"

        plt.plot(range(1854, len(past) + 1854), past, "-b", label=label1)
        plt.plot(range(len(past) + 1853, len(past) + len(pred) + 1853), pred, line[k], label=label[k])

    # plt.ylim(-1.8, -1.7)
    plt.xlim(1855, )
    plt.xlabel("Date/year")
    plt.ylabel("SST/degrees")
    plt.legend(loc='best')
    plt.title("SST over the years and prediction")
    plt.show()


if __name__ == '__main__':
    loadData()
