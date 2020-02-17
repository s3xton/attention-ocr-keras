import sys
sys.path.insert(1, 'model')
from model import ReaderModel
import tensorflow as tf
import tensorflow_addons as tfa



class Trainer(object):
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_history = []
        chars = 'abcdefghijklmnopqrstuvwxyz'
        charset = dict(enumerate(chars))
        self.model = ReaderModel(input_shape=(240, 240, 3),
                                 seq_length=8,
                                 rnn_size=10,
                                 charset=charset)

    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            chars_logit = self.model((images, None)).chars_logit
            print('Model output')
            # Add asserts to check the shape of the output.
            # tf.debugging.assert_equal(chars_logit.shape, (32, 20, 26))

            loss_value = self.model.create_loss(labels, chars_logit)

        self.loss_history.append(loss_value.numpy().mean())
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

    def train(self, epochs, dataset):
        for epoch in range(epochs):
            for (batch, (images, labels)) in enumerate(dataset):
                self.train_step(images, labels)
            print('Epoch {} finished'.format(epoch))
