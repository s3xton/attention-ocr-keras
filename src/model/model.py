from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from rnn import ChaRNN


class ReaderModel(tf.keras.Model):
    def __init__(self, input_shape, seq_length, rnn_size, num_char_classes, is_training=True):
        super(ReaderModel, self).__init__()
        self.feature_enc = tf.keras.applications.InceptionResNetV2(
            weights='imagenet', input_shape=input_shape, include_top=False)
        self.feature_enc.outputs = [
            self.feature_enc.get_layer('mixed_6a').output]
        self.rnn = ChaRNN(rnn_size, seq_length, num_char_classes)

    @tf.function
    def call(self, x):
        print('Input shape: {}'.format(x.shape))
        f = self.feature_enc(x)
        print('Feature enc shape: {}'.format(f.shape))
        f_enc = self.encode_coords(f)
        print('Feature enc with OH coords shape: {}'.format(f_enc.shape))
        f_pool = self.pool_views(f_enc)
        print('Pooled feature enc shape: {}'.format(f_pool.shape))
        logits, state = self.rnn((f_pool, None))
        print('Char logits shape: {}'.format(logits.shape))
        # predicted_chars, chars_log_prob, predicted_scores = (self.char_predictions(chars_logit))
        return logits

    @tf.function
    def encode_coords(self, f):
        """Adds one-hot encoding of coordinates to different views in the networks.
        For each "pixel" of a feature map it adds a onehot encoded x and y
        coordinates.
        Args:
        net: a tensor of shape=[batch_size, height, width, num_features]
        Returns:
        a tensor with the same height and width, but altered feature_size.
        """
        batch_size, h, w, _ = f.shape
        x, y = tf.meshgrid(tf.range(w), tf.range(h))
        w_loc = tf.one_hot(x, depth=w)
        h_loc = tf.one_hot(y, depth=h)
        loc = tf.concat([h_loc, w_loc], 2)
        loc = tf.tile(tf.expand_dims(loc, 0), [batch_size, 1, 1, 1])
        return tf.concat([f, loc], 3)

    @tf.function
    def pool_views(self, f_enc):
        """Combines output of multiple convolutional towers into a single tensor.
        It stacks towers one on top another (in height dim) in a 4x1 grid.
        The order is arbitrary design choice and shouldn't matter much.
        Args:
        nets: list of tensors of shape=[batch_size, height, width, num_features].
        Returns:
        A tensor of shape [batch_size, seq_length, features_size].
        """
        batch_size = f_enc.shape[0]
        feature_size = f_enc.shape[3]
        return tf.reshape(f_enc, [batch_size, -1, feature_size])

    @tf.function
    def char_predictions(self, chars_logit):
        """Returns confidence scores (softmax values) for predicted characters.
        Args:
        chars_logit: chars logits, a tensor with shape
            [batch_size x seq_length x num_char_classes]
        Returns:
        A tuple (ids, log_prob, scores), where:
            ids - predicted characters, a int32 tensor with shape
            [batch_size x seq_length];
            log_prob - a log probability of all characters, a float tensor with
            shape [batch_size, seq_length, num_char_classes];
            scores - corresponding confidence scores for characters, a float
            tensor
            with shape [batch_size x seq_length].
        """
        log_prob = utils.logits_to_log_prob(chars_logit)
        ids = tf.to_int32(tf.argmax(log_prob, axis=2), name='predicted_chars')
        mask = tf.cast(
        slim.one_hot_encoding(ids, self._params.num_char_classes), tf.bool)
        all_scores = tf.nn.softmax(chars_logit)
        selected_scores = tf.boolean_mask(all_scores, mask, name='char_scores')
        scores = tf.reshape(selected_scores, shape=(-1, self._params.seq_length))
        return ids, log_prob, scores