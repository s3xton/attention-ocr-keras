from trainer import Trainer
from data_source import DataSource
from model import OCRModel
import tensorflow as tf
import utils

tf.random.set_seed(54321)

rnn_size = 256
batch_size = 16

data_source = DataSource()
model = OCRModel(input_shape=data_source.config['image_shape'],
                 seq_length=data_source.config['max_sequence_length'],
                 rnn_size=rnn_size,
                 charset=data_source.charset,
                 num_views=data_source.config['num_of_views'])
trainer = Trainer(model, data_source.config['null_code'])

train_set = data_source.sets['train'].batch(batch_size)
num_batches = int(data_source.config['splits']['train']['size'] / batch_size)
trainer.train(25, train_set, num_batches)
