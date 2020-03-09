import sys
import re
import logging
import tensorflow as tf
from os.path import join, dirname


DEFAULT_DATASET_DIR = join(dirname(dirname(__file__)), 'data/fsns')

# The dataset configuration, should be used only as a default value.
DEFAULT_CONFIG = {
    'name': 'FSNS',
    'splits': {
        'train': {
            'size': 1974,  # 1044868,
            'pattern': 'train/train*'
        },
        # 'test': {
        #     'size': 20404,
        #     'pattern': 'test/test*'
        # },
        'validation': {
            'size': 252, #16150,
            'pattern': 'validation/validation*'
        }
    },
    'charset_filename': 'charset_size=134.txt',
    'image_shape': (150, 150, 3),
    'num_of_views': 4,
    'max_sequence_length': 37,
    'null_code': 133,
    'items_to_descriptions': {
        'image': 'A [150 x 600 x 3] color image.',
        'label': 'Characters codes.',
        'text': 'A unicode string.',
        'length': 'A length of the encoded text.',
        'num_of_views': 'A number of different views stored within the image.'
    }
}


class DataSource():

    def __init__(self, config=DEFAULT_CONFIG, dataset_dir=DEFAULT_DATASET_DIR):
        self.config = config
        self._dataset_dir = dataset_dir
        self.sets = self._load_splits(config['splits'])
        self.charset = self._parse_charset(
            join(dataset_dir, config['charset_filename']))

    def _load_splits(self, splits):
        sets = {}
        for key, body in splits.items():
            pattern = join(self._dataset_dir, body['pattern'])
            print(pattern)
            filenames = tf.data.TFRecordDataset.list_files(pattern)
            dataset = tf.data.TFRecordDataset(filenames=filenames)
            sets[key] = dataset.map(self._process_example)
        return sets

    def _process_example(self, eg):
        example = tf.io.parse_example(
            eg[tf.newaxis], {
                'image/encoded': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
                'image/text': tf.io.FixedLenFeature(shape=(), dtype=tf.string)
            })
        image = tf.io.decode_png(example['image/encoded'][0])
        return tf.cast(image, dtype=tf.float32), example['image/text'][0]

    def _parse_charset(self, charset_file, null_character=u'\u2591'):
        pattern = re.compile(r'(\d+)\t(.+)')
        charset = {}
        with open(charset_file) as fh:
            for i, line in enumerate(fh):
                m = pattern.match(line)
                if m is None:
                    logging.warning(
                        'incorrect charset file. line #%d: %s', i, line)
                    continue
                code = int(m.group(1))
                char = m.group(2)
                if char == '<nul>':
                    char = null_character
                charset[code] = char
        return charset
