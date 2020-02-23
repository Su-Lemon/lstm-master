import os
import os.path as osp

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
FLAGS2 = {}
FLAGS3 = [12, 17, 170, 175]  # [12, 17, 170, 175] [0, 2, 10, 17]  # 海表温度的预测区域

# For optimizer
tf.app.flags.DEFINE_float('learning_rate', 0.04, "Learning rate")
tf.app.flags.DEFINE_float('lambda_l2_reg', 0.003, "L2 regularization of weight")

# For net
tf.app.flags.DEFINE_integer('seq_len', 60, "Input and output sequence length")
# tf.app.flags.DEFINE_integer('test_seq_len', 120, "Sequence length")
tf.app.flags.DEFINE_integer('output_dim', 25, "Number of sequences to predict")
tf.app.flags.DEFINE_integer('input_dim', 25, "Number of sequences to predict")
tf.app.flags.DEFINE_integer('hidden_dim', 12, "LSTM(GRU) hidden dim")
tf.app.flags.DEFINE_integer('layers_num', 2, "LSTMCell(GRUCell) num")
tf.app.flags.DEFINE_integer('seq_num', 25, "LSTMCell(GRUCell) num")


# For training
tf.app.flags.DEFINE_integer('batch_size', 8, "Network batch size during training")
tf.app.flags.DEFINE_integer('max_iters', 20000, "Max iteration")
tf.app.flags.DEFINE_string('model_dir', "./models/", "Pretrained network weights")

tf.app.flags.DEFINE_integer('test_size', 1, "Number of testing")

# For data
FLAGS2['stockDataRoot'] = "./data/stock.csv"
FLAGS2['headers'] = ['date', 'code', 'name', 'Close', 'top_price', 'low_price',
                     'opening_price', 'bef_price', 'floor_price', 'floor',
                     'exchange', 'volume', 'amount', 'all_value', 'flow_value',
                     'none']
FLAGS2['predictor'] = ['Close']

FLAGS2['sst'] = ['last', 'pred']
FLAGS2['sstDataRoot'] = "./data/ERsst.nc"

FLAGS2['choose'] = 'sst'  # stock or sst
