import sys

from timebudget import timebudget
import tensorflow_addons as tfa
import tensorflow as tf
from model import ReaderModel

import timeit
import datetime


class Trainer(object):
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adadelta()
        self.loss_history = []
        chars = '?abcdefghijklmnopqrstuvwxyz'
        charset = dict(enumerate(chars))
        self.model = ReaderModel(input_shape=(240, 240, 3),
                                 seq_length=8,
                                 rnn_size=256,
                                 charset=charset)
        self.writer = tf.summary.create_file_writer("logdir")

    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            chars_logit = self.model((images, labels)).chars_logit
            # Add asserts to check the shape of the output.
            # tf.debugging.assert_equal(chars_logit.shape, (32, 20, 26))

            loss_value, label_enc = self.model.loss(labels, chars_logit)

        # self.loss_history.append(loss_value.numpy().mean())
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        return tf.reduce_mean(loss_value), label_enc, chars_logit

    @timebudget
    def train(self, epochs, dataset):
        print('Beginning training...')
        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

            progbar = tf.keras.utils.Progbar(
                len(list(dataset.as_numpy_iterator())))

            # Training loop for each batch
            for (batch, (images, labels)) in enumerate(dataset):
                # Optimise the model
                loss_value, label_enc, chars_logit = self.train_step(
                    images, labels)

                # Record the training loss and accuracy
                epoch_loss_avg(loss_value)
                epoch_accuracy(label_enc, chars_logit)

                # Update the progress bar
                progbar.update(batch+1)

            #

            # Write metrics
            with self.writer.as_default():
                tf.summary.scalar(
                    'train_loss', epoch_loss_avg.result(), step=epoch)
                tf.summary.scalar(
                    'train_accuracy', epoch_accuracy.result(), step=epoch)
            self.writer.flush()

            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))

        print('Training complete.')
