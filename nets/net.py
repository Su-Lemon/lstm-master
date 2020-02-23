import tensorflow as tf
import numpy as np
import pandas as pd

from ..configs import config as cfg
pd.options.mode.chained_assignment = None  # default='warn'


class Net(object):
    def __init__(self):
        self.encoder_input = []
        self.expected_output = []
        self.decode_input = []
        self.losses = {}

        self.tcells = []
        self.Mcell = []
        self.reshaped_outputs = []

    def createNet(self, sess):
        for i in range(cfg.FLAGS.seq_len):
            self.encoder_input.append(
                tf.placeholder(tf.float32, shape=(None, cfg.FLAGS.input_dim)))
            self.expected_output.append(
                tf.placeholder(tf.float32, shape=(None, cfg.FLAGS.output_dim)))
            self.decode_input.append(
                tf.placeholder(tf.float32, shape=(None, cfg.FLAGS.input_dim)))

        # Create LSTM(GRU)
        for i in range(cfg.FLAGS.layers_num):
            self.tcells.append(tf.contrib.rnn.GRUCell(cfg.FLAGS.hidden_dim))
        self.Mcell = tf.contrib.rnn.MultiRNNCell(self.tcells)

        # Connected by Encoder-Decoder
        dec_outputs, dec_memory = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(
            self.encoder_input, self.decode_input, self.Mcell)

        # Create output leyer
        for ii in dec_outputs:
            self.reshaped_outputs.append(
                tf.contrib.layers.fully_connected(ii, cfg.FLAGS.output_dim,
                                                  activation_fn=None))
        # L2 loss
        output_loss = 0
        for _y, _Y in zip(self.reshaped_outputs, self.expected_output):
            output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))
        self.losses['output_loss'] = output_loss

        # generalization capacity
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if not ("fully_connected" in tf_var.name):
                # print(tf_var.name)
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
        self.losses['reg_loss'] = reg_loss

        loss = output_loss + cfg.FLAGS.lambda_l2_reg * reg_loss
        self.losses['loss'] = loss
        return self.losses

    def trainer(self, sess, dataset, train_op, isTrain):
        X, Y = dataset.generateData(isTrain=True)
        feed_dict = {self.encoder_input[t]: X[t] for t in range(len(self.encoder_input))}
        feed_dict.update({self.expected_output[t]: Y[t] for t in range(len(self.expected_output))})

        c = np.concatenate(([np.zeros_like(Y[0])], Y[:-1]), axis=0)
        feed_dict.update({self.decode_input[t]: c[t] for t in range(len(c))})

        if isTrain:
            _, loss_t = sess.run([train_op, self.losses['loss']], feed_dict)
            return loss_t
        else:
            output_lossv, reg_lossv, loss_t = sess.run(
                [self.losses['output_loss'], self.losses['reg_loss'],
                 self.losses['loss']],
                feed_dict)
            print("-----------------")
            print(output_lossv, reg_lossv)
            return loss_t


    def demo(self, sess, dataset):
        X, Y = dataset.generateData(isTrain=False)
        feed_dict = {self.encoder_input[t]: X[t] for t in range(cfg.FLAGS.seq_len)}
        c = np.concatenate(
            ([np.zeros_like(Y[0])], Y[0:cfg.FLAGS.seq_len - 1]), axis=0)
        feed_dict.update(
            {self.decode_input[t]: c[t] for t in range(len(c))})
        outputs = np.array(sess.run([self.reshaped_outputs], feed_dict)[0])
        return outputs
