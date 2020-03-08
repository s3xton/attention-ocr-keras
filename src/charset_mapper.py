import tensorflow as tf


class CharsetMapper(object):
    """A simple class to map tensor ids into strings.
    It works only when the character set is 1:1 mapping between individual
    characters and individual ids.
    Make sure you call tf.tables_initializer().run() as part of the init op.
    """

    def __init__(self, charset, max_sequence_length, default_character='?'):
        """Creates a lookup table.
        Args:
            charset: a dictionary with id-to-character mapping.
        """
        ids_tensor = tf.constant(list(charset.keys()), dtype=tf.int64)
        chars_tensor = tf.constant(list(charset.values()), dtype=tf.string)
        self.max_sequence_length = max_sequence_length
        self.table_ids = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(ids_tensor, chars_tensor, key_dtype=tf.int64, value_dtype=tf.string), default_character)

        self.table_text = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(chars_tensor, ids_tensor, key_dtype=tf.string, value_dtype=tf.int64), 0)

    def get_text(self, ids):
        """Returns a string corresponding to a sequence of character ids.
            Args:
                ids: a tensor with shape [batch_size, max_sequence_length]
            """
        return tf.strings.reduce_join(self.table_ids.lookup(tf.cast(ids, dtype=tf.int64)), axis=1)

    def get_ids(self, text):
        """Returns a sequence of character ids corresponding to a string.
            Args:
                text: a tensor with shape [batch_size,]
            """
        char_seq = tf.strings.bytes_split(text)
        char_seq = tf.concat(
            [char_seq, [['?'] * self.max_sequence_length]], axis=0)
        dense_seq = tf.sparse.to_dense(char_seq.to_sparse(), default_value='?')
        dense_seq = dense_seq[:-1, :]
        return self.table_text.lookup(dense_seq)
