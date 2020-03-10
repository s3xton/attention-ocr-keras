import sys
import time
import math
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from os.path import join, exists
from os import makedirs
from timebudget import timebudget
from tensorflow.python.framework.errors_impl import NotFoundError


class Trainer(object):

    def __init__(self, model, null_code, output_path='/tmp/attention-ocr-keras'):
        self.optimizer = tf.compat.v1.train.MomentumOptimizer(0.004, 0.9)
        self.loss_history = []
        self.model = model
        self.output_path = output_path
        self.null_code = null_code
        self.metrics = MetricsRecorder(['train', 'validation'], output_path)

        # Setup checkpointing and load if theres an existing one
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(
            checkpoint, directory=join(output_path, 'model'), max_to_keep=3)
        checkpoint.restore(self.manager.latest_checkpoint)


    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # Call the model with the ground truth as input for autoregression
            output_endpoint = self.model((images, labels))
            loss_value, label_enc = self.model.loss(
                labels, output_endpoint.chars_logit)

        # Apply the gradients
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        return output_endpoint.predicted_chars, label_enc, loss_value

    @timebudget
    def train(self, epochs, data_source, batch_size):
        try:
            self._train(epochs, data_source, batch_size)
        finally:
            self.metrics.complete()

    def _train(self, epochs, data_source, batch_size):
        # Treat the train set as one big repeating dataset
        train_set = data_source.sets['train'].batch(batch_size).repeat(epochs)
        num_batches = math.ceil(
            data_source.config['splits']['train']['size'] / batch_size)
        max_global_step = num_batches * epochs

        progbar = tf.keras.utils.Progbar(
            max_global_step, stateful_metrics=['train/loss', 'train/accuracy'])

        print('Beginning training.')
        # Training loop for each batch
        for (step, (images, labels)) in enumerate(train_set):
            # Run a single step
            x, y, loss = self.train_step(images, labels)
            
            # Record metrics
            self.metrics.record(split='train', x=x, y=y, loss=loss, step=step)
            progbar.update(step+1, values=[('train/loss', loss)])
            
            # Save the model
            if step % 5 == 0:
                self.manager.save(checkpoint_number=step)
            
        print('Training complete.')
        
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


class MetricsRecorder():

    # TODO properly implement splits in the metrics
    def __init__(self, splits, output_path):
        self.output_path = output_path
        self.history_path = join(self.output_path, 'history.csv')
        if not exists(output_path):
            makedirs(output_path)
        # TODO test/fix history loading and saving
        # if exists(self.history_path):
        #     self.history = pd.read_csv(self.history_path).to_records()
        # else:
        #     self.history = []
        self.history = []
        self.writer = tf.summary.create_file_writer(join(output_path, 'summary'))
        

    def record(self, split, x, y, loss, step):
        history_step = [('step', step), ('loss', loss)]
        
        with self.writer.as_default():
            tf.summary.scalar(
                '{}/loss'.format(split), loss, step=step)
        self.writer.flush()

        self.history.append(history_step)

    def complete(self):
        history = pd.DataFrame.from_records(self.history)
        history.to_csv(self.history_path, index=False)
        self.writer.close()

