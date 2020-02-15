from trainer import Trainer
import tensorflow as tf
import utils

trainer = Trainer()

train_set = tf.data.Dataset.list_files(str('../data/simple_images/train/*'))
train_set = train_set.map(utils.process_path, num_parallel_calls=5)

for image, label in train_set.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())

# trainer.train(1, train_set.batch(2))