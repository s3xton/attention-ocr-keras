import sys
import time
import math
import tensorflow as tf
import numpy as np
from datetime import datetime
from os.path import join
from timebudget import timebudget


class Trainer(object):
    def __init__(self, model, null_code, checkpoint_file='./checkpoint'):
        # self.optimizer = tf.compat.v1.train.MomentumOptimizer(0.004, 0.9)
        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.001)
        self.loss_history = []
        self.model = model
        self.checkpoint_file = checkpoint_file
        self.null_code = null_code
        ts = time.time()
        self.writer = tf.summary.create_file_writer(
            join("logdir", datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')))

    def eval(self, val_set, num_batches, val_loss_avg, val_accuracy):
        print('Validating.')

        progbar = tf.keras.utils.Progbar(
                num_batches, stateful_metrics=['val/loss', 'val/accuracy'])

        for (batch, (images, labels)) in enumerate(val_set):
            output_endpoint = self.model((images, None))
            loss_value, label_enc = self.model.loss(
                labels, output_endpoint.chars_logit)

            val_loss_avg(loss_value)
            mask = self._get_mask(label_enc)
            val_accuracy(label_enc, output_endpoint.predicted_chars,
                         sample_weight=mask)
            
            print(output_endpoint.predicted_chars)

            progbar.update(
                    batch+1, values=[('val/loss', loss_value), ('val/accuracy', val_accuracy.result())])

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
    def train(self, epochs, data_source, batch_size):
        tf.summary.trace_on()
        num_batches = math.ceil(
            data_source.config['splits']['train']['size'] / batch_size)
        best_loss = np.inf
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.Accuracy()
        val_loss_avg = tf.keras.metrics.Mean()
        val_accuracy = tf.keras.metrics.Accuracy()
        train_set = data_source.sets['train'].batch(batch_size)

        print('Beginning training.')
        for epoch in range(epochs):
            progbar = tf.keras.utils.Progbar(
                num_batches, stateful_metrics=['train/loss', 'train/accuracy'])

            # Training loop for each batch
            for (batch, (images, labels)) in enumerate(train_set):
                # Optimise the model
                loss_value, label_enc, predicted_chars = self.train_step(
                    images, labels)

                # Record the training loss and accuracy
                epoch_loss_avg(loss_value)
                mask = self._get_mask(label_enc)
                epoch_accuracy(label_enc, predicted_chars, sample_weight=mask)

                # Update the progress bar
                progbar.update(
                    batch+1, values=[('train/loss', loss_value), ('train/accuracy', epoch_accuracy.result())])

            # Evaluate the model
            val_batches = math.ceil(
                data_source.config['splits']['validation']['size'] / batch_size)
            self.eval(data_source.sets['validation'].batch(batch_size), val_batches, val_loss_avg, val_accuracy)

            # Write metrics
            with self.writer.as_default():
                tf.summary.scalar(
                    'train/loss', epoch_loss_avg.result(), step=epoch)
                tf.summary.scalar(
                    'train/accuracy', epoch_accuracy.result(), step=epoch)
                tf.summary.scalar(
                    'val/loss', val_loss_avg.result(), step=epoch)
                tf.summary.scalar(
                    'val/accuracy', val_accuracy.result(), step=epoch)
            self.writer.flush()

            log = "Epoch {:03d}: train/loss: {:.3f}, train/accuracy: {:.3%}, val/loss: {:.3f}, val/accuracy: {:.3%}"
            print(log.format(epoch,
                             epoch_loss_avg.result(),
                             epoch_accuracy.result(),
                             val_loss_avg.result(),
                             val_accuracy.result()))
            
            new_loss = val_loss_avg.result().numpy()
            if new_loss < best_loss:
                print("Saving new model.")
                self.model.save_weights(self.checkpoint_file, overwrite=True)
                best_loss = new_loss

            # Reset the metrics between epochs
            epoch_loss_avg.reset_states()
            epoch_accuracy.reset_states()
            val_loss_avg.reset_states()
            val_accuracy.reset_states()

        with self.writer.as_default():
            tf.summary.trace_export("graph", step=0)

        print('Training complete.')
