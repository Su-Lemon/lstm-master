# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import netCDF4 as nc

from ..configs import config as cfg
pd.options.mode.chained_assignment = None


class SeqData(object):
    def __init__(self):
        self.Y_train = []
        self.X_train = []
        self.X_test = []
        self.Y_test = []

        self.yearSST = []
        self.dataMinMax = []

        self.dataCache = self.load()

    def load(self):
        if cfg.FLAGS2['choose'] == 'sst':
            # SST prediction
            file = nc.Dataset(cfg.FLAGS2['sstDataRoot'])
            print('Meteorological data includes:\n', file.variables.keys())
            fileTime = file.variables['time'][:]
            print('Observation time：', fileTime.shape)
            fileSST = file.variables['sst'][:]
            print('SST map：', fileSST.shape)
            used = fileSST[:, cfg.FLAGS3[0]:cfg.FLAGS3[1], cfg.FLAGS3[2]:cfg.FLAGS3[3]]
            print("Area used：", used.shape)

            # Convert monthly average temperature to annual average temperature
            temp = 0
            everyLoc = []
            for row in range(cfg.FLAGS3[1] - cfg.FLAGS3[0]):
                for col in range(cfg.FLAGS3[3] - cfg.FLAGS3[2]):
                    for val in range(len(fileTime)):
                        if val % 12 == 0:
                            everyLoc.append(temp / 12)
                            temp = 0
                        temp += used[val, row, col]
                    self.yearSST.append(everyLoc)
                    everyLoc = []
            self.dataMinMax.append(min(min(row) for row in self.yearSST))
            self.dataMinMax.append(max(max(row) for row in self.yearSST))
            print("All seqs", len(self.yearSST), "All years", len(self.yearSST[0]))
            return fileSST
        elif cfg.FLAGS2['choose'] == 'stock':
            # Stock forecast
            data = pd.read_csv(cfg.FLAGS2['dataRoot'], names=cfg.FLAGS2['headers'], header=None,encoding="gbk")
            return data

    def loadData(self, window_size):
        if cfg.FLAGS2['choose'] == 'sst':
            array = np.asarray(self.yearSST)
            kept_values = array[:]
            kept_values = (kept_values - self.dataMinMax[0]) / (self.dataMinMax[1] - self.dataMinMax[0])
            kept_values = kept_values.transpose()
        elif cfg.FLAGS2['choose'] == 'stock':
            array = np.asarray(self.dataCache[cfg.FLAGS2['predictor']].values)
            kept_values = array[:]

        X = []
        Y = []
        print("len(kept_values)", len(kept_values))
        for i in range(len(kept_values) - window_size * 2):
            X.append(kept_values[i:i + window_size])
            Y.append(kept_values[i + window_size:i + window_size * 2])

        X = np.reshape(X, [-1, window_size, cfg.FLAGS.seq_num])
        Y = np.reshape(Y, [-1, window_size, cfg.FLAGS.seq_num])
        print("Loaded raw data information\n（Seq num，Seq len，Seq out dim/type）", np.shape(X))
        return X, Y

    def generateData(self, isTrain):
        # First load, with memoization:
        if isTrain and len(self.Y_train) == 0:
            X, Y = self.loadData(window_size=cfg.FLAGS.seq_len)
            # Split 80-20:
            self.X_train = X[:int(len(X) * 0.8)]
            self.Y_train = Y[:int(len(Y) * 0.8)]
            # self.X_train = X
            # self.Y_train = Y
        elif not isTrain and len(self.Y_test) == 0:
            X, Y = self.loadData(window_size=cfg.FLAGS.seq_len)
            # Split 80-20:
            self.X_test = X[int(len(X) * 0.8):]
            self.Y_test = Y[int(len(Y) * 0.8):]

        if isTrain:
            return self.doGenerate(self.X_train, self.Y_train, cfg.FLAGS.batch_size, isTrain)
        else:
            return self.doGenerate(self.X_test, self.Y_test, cfg.FLAGS.test_size, isTrain)

    def doGenerate(self, X, Y, size, isTrain):
        assert X.shape == Y.shape, (X.shape, Y.shape)
        if isTrain:
            idxes = np.random.randint(X.shape[0], size=size)
            X_out = np.array(X[idxes]).transpose((1, 0, 2))
            Y_out = np.array(Y[idxes]).transpose((1, 0, 2))
        elif not isTrain:
            idxes = np.array([X.shape[0]-1]*size)
            X_out = np.array(Y[idxes]).transpose((1, 0, 2))
            Y_out = np.array(Y[idxes]).transpose((1, 0, 2))

        return X_out, Y_out


if __name__ == '__main__':
    pass
