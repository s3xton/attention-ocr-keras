from trainer import Trainer
import tensorflow as tf
import utils

tf.random.set_seed(1)

trainer = Trainer()

train_set = tf.data.Dataset.list_files(str('data/simple_images/train/*'))
train_set = train_set.map(utils.process_path, num_parallel_calls=5)

# for image, label in train_set.take(1):
#   print("Image shape: ", image.numpy().shape)
#   print("Label: ", label.numpy())

train_set = train_set.batch(32)
trainer.train(25, train_set)