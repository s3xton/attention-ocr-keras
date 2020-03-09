import math
import argparse
import tensorflow as tf
from trainer import Trainer
from data import DataSource
from model import OCRModel

tf.random.set_seed(54321)
rnn_size = 256
batch_size = 16

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default=None, type=str,
                    help="root directory of the dataset files")
args = parser.parse_args()
if args.dataset_dir is not None:
    data_source = DataSource(dataset_dir=args.dataset_dir)
else:
    data_source = DataSource()

model = OCRModel(input_shape=data_source.config['image_shape'],
                 seq_length=data_source.config['max_sequence_length'],
                 rnn_size=rnn_size,
                 charset=data_source.charset,
                 num_views=data_source.config['num_of_views'])
trainer = Trainer(model, data_source.config['null_code'])
trainer.train(25, data_source, batch_size)
