from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from rnn import ChaRNN


class ReaderModel(tf.keras.Model):
    def __init__(self, rnn_size):
        super(ReaderModel, self).__init__()
        self.feature_enc = tf.keras.applications.InceptionResNetV2(weights='imagenet', input_shape=(75,75,3), include_top=False)
        self.feature_enc.outputs = [self.feature_enc.get_layer('mixed_6a').output]
        self.rnn = ChaRNN(rnn_size)

    def call(self, x):
        x = self.feature_enc(x)
        # TODO need alpha of attention layer
        x, _ = self.rnn(x)
        return x


    