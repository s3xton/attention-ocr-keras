from tensorflow.keras import layers
import tensorflow as tf
import sys

class ChaRNN(layers.Layer):
    '''
    This class implements a custom layer with an LSTMCell to allow autocorelated links 
    from each steps output to input
    '''

    def __init__(self, rnn_size, seq_length, num_char_classes):
        super(ChaRNN, self).__init__()
        self.rnn_size = rnn_size
        self.cell = layers.LSTMCell((num_char_classes, rnn_size))
        self.seq_length = seq_length
        self._char_logits = {}
        self.num_char_classes = num_char_classes
        w_init = tf.random_normal_initializer()
        self._softmax_w = tf.Variable(initial_value=w_init(shape=(rnn_size, num_char_classes),
                                              dtype='float32'),
                                      trainable=True)
        self._softmax_b = tf.Variable(initial_value=w_init(shape=(rnn_size, num_char_classes),
                                              dtype='float32'),
                                      trainable=True)

    @tf.function
    def call(self, inputs):
        print('RNN')
        f_pool, ground_truth = inputs
        batch_size, timestep, _ = f_pool.shape
        # Variable for collecting all outputs
        outputs = []
        
        # Initial state and previous output
        state = tf.zeros((batch_size, self.rnn_size), dtype=tf.float32)
        prev_output = tf.zeros([batch_size, self.num_char_classes])

        # for i in range(timestep):
        for i in tf.range(1):
            print(prev_output.shape)
            output, state = self.cell(prev_output, state)
            print(output.shape)
            outputs = outputs.append(output)
            if ground_truth is not None:
                prev_output = ground_truth[:, i, :]
            else:
                prev_output = self.char_logit(output, i)
                prev_output = self.char_one_hot(prev_output)


        return tf.transpose(outputs.stack(), [1, 0, 2]), state

    @tf.function
    def get_image_feature(self, f_enc, char_index):
        """Returns a subset of image features for a character.
        Args:
        char_index: an index of a character.
        Returns:
        A tensor with shape [batch_size, ?]. The output depth depends on the
        depth of input net.
        """
        batch_size, features_num, _ = f_enc.shape
        slice_len = int(features_num / self.seq_length)
        # In case when features_num != seq_length, we just pick a subset of image
        # features, this choice is arbitrary and there is no intuitive geometrical
        # interpretation. If features_num is not dividable by seq_length there will
        # be unused image features.
        f_slice = f_enc[:, char_index:char_index + slice_len, :]
        feature = tf.reshape(f_slice, [batch_size, -1])
        return feature

    @tf.function
    def char_logit(self, inputs, char_index):
        """Creates logits for a character if required.
        Args:
        inputs: A tensor with shape [batch_size, ?] (depth is implementation
            dependent).
        char_index: A integer index of a character in the output sequence.
        Returns:
        A tensor with shape [batch_size, num_char_classes]
        """
        if char_index not in self._char_logits:
            self._char_logits[char_index] = inputs * self._softmax_w + self._softmax_b
        return self._char_logits[char_index]

    @tf.function
    def char_one_hot(self, logit):
        """Creates one hot encoding for a logit of a character.
        Args:
        logit: A tensor with shape [batch_size, num_char_classes].
        Returns:
        A tensor with shape [batch_size, num_char_classes]
        """
        prediction = tf.argmax(logit, axis=1)
        return slim.one_hot_encoding(prediction, self._params.num_char_classes)


