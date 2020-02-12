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
        self.num_char_classes = num_char_classes
        self.seq_length = tf.constant(seq_length)
        self.cell = layers.LSTMCell(self.rnn_size)
        self.dense = layers.Dense(self.num_char_classes)
        self._char_logits = {}
        
        w_init = tf.random_normal_initializer()
        self._softmax_w = tf.Variable(initial_value=w_init(shape=(self.rnn_size, self.num_char_classes),
                                              dtype='float32'),
                                      trainable=True)
        self._softmax_b = tf.Variable(initial_value=w_init(shape=(self.num_char_classes,),
                                              dtype='float32'),
                                      trainable=True)

    @tf.function
    def call(self, inputs):
        f_pool, ground_truth = inputs
        batch_size, _, _ = f_pool.shape
        outputs = tf.TensorArray(tf.float32, size=self.seq_length)
        
        # Initial state and previous output
        prev_output = tf.zeros([batch_size, self.num_char_classes])
        state = [tf.zeros([batch_size, self.rnn_size])] * 2
        

        for t in tf.range(self.seq_length):
            # Concat previous output and the image feature to create the input
            image_feature = self.get_image_feature(f_pool, t)
            x_t = tf.concat([image_feature, prev_output], axis=1)
            
            # Run through the cell
            output, state = self.cell(x_t, state)

            # Softmax for the logits then OH encode
            output = self.char_logit(output)
            output = self.char_one_hot(output)
            outputs.write(t, output)

            # If training, ground truth will be passed and should be used
            # for autoregression, for inference use actual network out
            if ground_truth is not None:
                prev_output = ground_truth[:, t, :]
            else:
                prev_output = output
            
        sequence_output = tf.transpose(outputs.stack(), [1, 0, 2])
        print(sequence_output.shape)
        return sequence_output, state

    @tf.function
    def get_image_feature(self, f_pool, char_index):
        """Returns a subset of image features for a character.
        Args:
        char_index: an index of a character.
        Returns:
        A tensor with shape [batch_size, ?]. The output depth depends on the
        depth of input net.
        """
        
        batch_size, features_num, _ = f_pool.shape
        slice_len = int(features_num / self.seq_length)
        # In case when features_num != seq_length, we just pick a subset of image
        # features, this choice is arbitrary and there is no intuitive geometrical
        # interpretation. If features_num is not dividable by seq_length there will
        f_slice = tf.slice(f_pool, [0, char_index, 0], [-1, slice_len, -1])
        feature = tf.reshape(f_slice, [batch_size, -1])
        return feature

    @tf.function
    def char_logit(self, inputs):
        """Creates logits for a character if required.
        Args:
        inputs: A tensor with shape [batch_size, ?] (depth is implementation
            dependent).
        char_index: A integer index of a character in the output sequence.
        Returns:
        A tensor with shape [batch_size, num_char_classes]
        """
        # if char_index not in self._char_logits:
        #     self._char_logits[char_index] = inputs * self._softmax_w + self._softmax_b
        return tf.matmul(inputs, self._softmax_w) + self._softmax_b

    @tf.function
    def char_one_hot(self, logit):
        """Creates one hot encoding for a logit of a character.
        Args:
        logit: A tensor with shape [batch_size, num_char_classes].
        Returns:
        A tensor with shape [batch_size, num_char_classes]
        """
        prediction = tf.argmax(logit, axis=1)
        return tf.one_hot(prediction, depth=self.num_char_classes)


