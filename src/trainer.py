import sys

from timebudget import timebudget
import tensorflow_addons as tfa
import tensorflow as tf
from os.path import join
import time
from datetime import datetime


class Trainer(object):
    def __init__(self, model, null_code):
        # self.optimizer = tf.compat.v1.train.MomentumOptimizer(0.004, 0.9)
        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.001)
        self.loss_history = []
        self.model = model
        self.null_code = null_code
        ts = time.time()
        self.writer = tf.summary.create_file_writer(
            join("logdir", datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')))

    def _get_mask(self, labels):
        # Suppose that reject character is always 0.
        reject_char = tf.constant(
            self.null_code,
            shape=labels.shape,
            dtype=tf.int64)
        known_char = tf.not_equal(labels, reject_char)
        mask = tf.cast(known_char, dtype=tf.int32)
        return mask

    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            output_endpoint = self.model((images, labels))

            # Add asserts to check the shape of the output.
            # tf.debugging.assert_equal(chars_logit.shape, (32, 20, 26))

            loss_value, label_enc = self.model.loss(
                labels, output_endpoint.chars_logit)

        self.loss_history.append(loss_value.numpy().mean())
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        return loss_value, label_enc, output_endpoint.predicted_chars

    @timebudget
    def train(self, epochs, dataset, num_batches):
        tf.summary.trace_on()
        print('Beginning training...')
        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.Accuracy()

            progbar = tf.keras.utils.Progbar(num_batches)

            # Training loop for each batch
            for (batch, (images, labels)) in enumerate(dataset):
                # Optimise the model
                loss_value, label_enc, predicted_chars = self.train_step(
                    images, labels)

                # Record the training loss and accuracy
                epoch_loss_avg(loss_value)
                mask = self._get_mask(label_enc)
                epoch_accuracy(label_enc, predicted_chars, sample_weight=mask)

                # Update the progress bar
                progbar.update(batch+1)

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
        with self.writer.as_default():
            tf.summary.trace_export("graph", step=0)

        print('Training complete.')
