import tensorflow as tf
import numpy as np
import pandas as pd

from lstm_master.datasets.dataset import SeqData
from lstm_master.nets.net import Net
from lstm_master.configs import config as cfg
from lstm_master.utils.draw import plotLoss
pd.options.mode.chained_assignment = None


class Train(object):
    def __init__(self):
        self.dataSet = SeqData()
        self.net = Net()
        self.train_losses = []
        self.test_losses = []

    def training(self):
        tf.reset_default_graph()
        sess = tf.InteractiveSession()

        losses = self.net.createNet(sess)
        train_op = tf.train.AdamOptimizer(cfg.FLAGS.learning_rate).minimize(losses['loss'])

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for t in range(cfg.FLAGS.max_iters + 1):
            train_loss = self.net.trainer(sess, self.dataSet, train_op, True)
            self.train_losses.append(train_loss)
            if t % 50 == 0:
                test_loss = self.net.trainer(sess, self.dataSet, train_op, False)
                self.test_losses.append(test_loss)
                print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t, cfg.FLAGS.max_iters, train_loss, test_loss))
        print("Fin. train loss: {}, \tTEST loss: {}".format(train_loss, test_loss))
        save_path = saver.save(sess, cfg.FLAGS.model_dir + "lstm.ckpt")
        print("Model saved in file: %s" % save_path)
        plotLoss(self.test_losses, self.train_losses)


if __name__ == '__main__':
    train = Train()
    train.training()
