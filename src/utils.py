import tensorflow as tf
import os

def logits_to_log_prob(logits):
    """Computes log probabilities using numerically stable trick.
    This uses two numerical stability tricks:
    1) softmax(x) = softmax(x - c) where c is a constant applied to all
    arguments. If we set c = max(x) then the softmax is more numerically
    stable.
    2) log softmax(x) is not numerically stable, but we can stabilize it
    by using the identity log softmax(x) = x - log sum exp(x)
    Args:
        logits: Tensor of arbitrary shape whose last dimension contains logits.
    Returns:
        A tensor of the same shape as the input, but with corresponding log
        probabilities.
    """

    reduction_indices = len(logits.shape.as_list()) - 1
    max_logits = tf.math.reduce_max(logits, axis=reduction_indices, keepdims=True)
    safe_logits = tf.subtract(logits, max_logits)
    sum_exp = tf.math.reduce_sum(tf.exp(safe_logits), axis=reduction_indices,keepdims=True)
    log_probs = tf.math.subtract(safe_logits, tf.math.log(sum_exp))
    return log_probs

def get_label(file_path, max_seq_length):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  label = tf.strings.split(parts[-1], '.')[0]
  # The second to last is the class-directory
  return label.ljust(max_seq_length, '?')

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [240, 240])

def process_path(file_path, max_seq_length=8):
  label = get_label(file_path, max_seq_length)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label