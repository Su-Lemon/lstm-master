import tensorflow as tf
import numpy as np
import pandas as pd

from lstm_master.datasets.dataset import SeqData
from lstm_master.nets.net import Net
from lstm_master.configs import config as cfg
from lstm_master.utils.draw import plotPred, plotSSTMap
pd.options.mode.chained_assignment = None


class Test(object):
    def __init__(self):
        self.dataSet = SeqData()
        self.net = Net()
        self.train_losses = []
        self.test_losses = []

    def testing(self):
        print("visualize {} predictions data:".format(cfg.FLAGS.test_size))
        lstm_model = tf.train.latest_checkpoint(cfg.FLAGS.model_dir)
        sess = tf.InteractiveSession()
        self.net.createNet(sess)
        saver = tf.train.Saver()
        saver.restore(sess, lstm_model)

        preout = self.net.demo(sess, self.dataSet)
        preout = np.reshape(preout, [cfg.FLAGS.seq_len, cfg.FLAGS.test_size, cfg.FLAGS.output_dim])
        print("reshapenp.shape(preout):", np.shape(preout))
        preout = preout*(self.dataSet.dataMinMax[1]-self.dataSet.dataMinMax[0])+self.dataSet.dataMinMax[0]

        plotSSTMap(preout, self.dataSet.dataCache)
        plotPred(preout, self.dataSet.yearSST)


if __name__ == '__main__':
    test = Test()
    test.testing()
