from tensorflow.keras import layers
import tensorflow as tf

class ChaRNN(layers.Layer):

    def __init__(self, rnn_size):
        super(ChaRNN, self).__init__()
        self.rnn_size = rnn_size
        self.cell = layers.LSTMCell(rnn_size)


    @tf.function
    def call(self, input_data):
        outputs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        state = tf.zeros((input_data.shape[0], self.rnn_size), dtype=tf.float32)

        for i in tf.range(input_data.shape[1]):
            print(input)
            output, state = self.cell(tf.expand_dims(input_data[:, i, :], 1), state)
            print(output.shape)
            outputs = outputs.write(i, output)
        return tf.transpose(outputs.stack(), [1, 0, 2]), state


