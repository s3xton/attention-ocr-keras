from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import tensorflow as tf
import tensorflow_addons as tfa
import utils
from model_parameters import default_mparams, OutputEndpoints
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from charset_mapper import CharsetMapper
from rnn import ChaRNN


class ReaderModel(tf.keras.Model):
    def __init__(self, input_shape, seq_length, rnn_size, charset):
        super(ReaderModel, self).__init__()
        self._mparams = default_mparams()
        self.num_char_classes = len(charset)
        self.seq_length = seq_length
        self.charset = charset

        self.feature_enc = tf.keras.applications.InceptionResNetV2(
            weights='imagenet', input_shape=input_shape, include_top=False)
        self.feature_enc.outputs = [
            self.feature_enc.get_layer('mixed_6a').output]
        self.feature_enc.trainable = False
        self.rnn = ChaRNN(rnn_size, self.seq_length, self.num_char_classes)
        self.character_mapper = CharsetMapper(self.charset, self.seq_length)


    def call(self, x):
        # Split out the ground truth. During inference this is None,
        # during training it contains labels used for autoregression
        input_image, ground_truth = x

        # Encode the image using resnet
        f = self.feature_enc(input_image)
        
        # Add the spacial coords
        f_enc = self.encode_coords(f)
        f_pool = self.pool_views(f_enc)
        
        # Generate the logits from the sequential model
        if ground_truth is not None:
            ground_truth = self.character_mapper.get_ids(ground_truth)
        chars_logit, _ = self.rnn((f_pool, ground_truth))
        # Interpret the logits
        predicted_chars, chars_log_prob, predicted_scores = (
            self.char_predictions(chars_logit))
        predicted_text = self.character_mapper.get_text(predicted_chars)

        return OutputEndpoints(
            chars_logit=chars_logit,
            chars_log_prob=chars_log_prob,
            predicted_chars=predicted_chars,
            predicted_scores=predicted_scores,
            predicted_text=predicted_text)

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
        ids = tf.cast(tf.argmax(log_prob, axis=2),
                      tf.int32, name='predicted_chars')
        mask = tf.cast(tf.one_hot(ids, self.num_char_classes), tf.bool)
        all_scores = tf.nn.softmax(chars_logit)
        selected_scores = tf.boolean_mask(all_scores, mask, name='char_scores')
        scores = tf.reshape(selected_scores, shape=(-1, self.seq_length))
        return ids, log_prob, scores


    def loss(self, labels, chars_logit):
        """Creates all losses required to train the model.
        Args:
        data: InputEndpoints namedtuple.
        endpoints: Model namedtuple.
        Returns:
        Total loss.
        """
        batch_size, _, _ = chars_logit.shape
        labels = self.character_mapper.get_ids(labels)
        labels = tf.one_hot(labels, depth=self.num_char_classes)
        return tf.nn.softmax_cross_entropy_with_logits(logits=chars_logit, labels=labels, axis=2), labels


    def sequence_loss_fn(self, chars_logits, chars_labels):
        """Loss function for char sequence.
        Depending on values of hyper parameters it applies label smoothing and can
        also ignore all null chars after the first one.
        Args:
        chars_logits: logits for predicted characters,
            shape=[batch_size, seq_length, num_char_classes];
        chars_labels: ground truth ids of characters,
            shape=[batch_size, seq_length];
        mparams: method hyper parameters.
        Returns:
        A Tensor with shape [batch_size] - the log-perplexity for each sequence.
        """
        mparams = self._mparams['sequence_loss_fn']
        if mparams.label_smoothing > 0:
            smoothed_one_hot_labels = self.label_smoothing_regularization(
                chars_labels, mparams.label_smoothing)
            # labels_list = tf.unstack(smoothed_one_hot_labels, axis=1)
            labels_list = smoothed_one_hot_labels
        else:
            # NOTE: in case of sparse softmax we are not using one-hot
            # encoding.
            # labels_list = tf.unstack(chars_labels, axis=1)
            labels_list = chars_labels

        batch_size, seq_length, _ = chars_logits.shape.as_list()
        if mparams.ignore_nulls:
            weights = tf.ones((batch_size, seq_length), dtype=tf.float32)
        else:
            # Suppose that reject character is the last in the charset.
            reject_char = tf.constant(
                self.num_char_classes - 1,
                shape=(batch_size, seq_length),
                dtype=tf.int64)
            known_char = tf.not_equal(chars_labels, reject_char)
            weights = tf.cast(known_char, dtype=tf.float32)

        # logits_list = tf.unstack(chars_logits, axis=1)
        # weights_list = tf.unstack(weights, axis=1)
        logits_list = chars_logits
        weights_list = weights
        loss = tfa.seq2seq.sequence_loss(
            logits_list,
            labels_list,
            weights_list,
            softmax_loss_function=self.get_softmax_loss_fn(
                mparams.label_smoothing),
            average_across_timesteps=mparams.average_across_timesteps)
        return loss

    def label_smoothing_regularization(self, chars_labels, weight=0.1):
        """Applies a label smoothing regularization.
        Uses the same method as in https://arxiv.org/abs/1512.00567.
        Args:
        chars_labels: ground truth ids of charactes,
            shape=[batch_size, seq_length];
        weight: label-smoothing regularization weight.
        Returns:
        A sensor with the same shape as the input.
        """
        one_hot_labels = tf.one_hot(
            chars_labels, depth=self.num_char_classes, axis=-1)
        pos_weight = 1.0 - weight
        neg_weight = weight / self.num_char_classes
        return one_hot_labels * pos_weight + neg_weight

    def get_softmax_loss_fn(self, label_smoothing):
        """Returns sparse or dense loss function depending on the label_smoothing.
            Args:
            label_smoothing: weight for label smoothing
            Returns:
            a function which takes labels and predictions as arguments and returns
            a softmax loss for the selected type of labels (sparse or dense).
            """
        if label_smoothing > 0:

            def loss_fn(labels, logits):
                return (tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels))
        else:

            def loss_fn(labels, logits):
                return tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels)

        return loss_fn
