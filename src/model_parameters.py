import collections

OutputEndpoints = collections.namedtuple('OutputEndpoints', [
    'chars_logit', 'chars_log_prob', 'predicted_chars', 'predicted_scores',
    'predicted_text'
])

# TODO: replace with tf.HParams when it is released.
ModelParams = collections.namedtuple('ModelParams', [
    'num_char_classes', 'seq_length', 'num_views', 'null_code'
])

ConvTowerParams = collections.namedtuple('ConvTowerParams', ['final_endpoint'])

SequenceLogitsParams = collections.namedtuple('SequenceLogitsParams', [
    'use_attention', 'use_autoregression', 'num_lstm_units', 'weight_decay',
    'lstm_state_clip_value'
])

SequenceLossParams = collections.namedtuple('SequenceLossParams', [
    'label_smoothing', 'ignore_nulls', 'average_across_timesteps'
])

EncodeCoordinatesParams = collections.namedtuple('EncodeCoordinatesParams', [
    'enabled'
])


def default_mparams():
    return {
        'conv_tower_fn':
        ConvTowerParams(final_endpoint='Mixed_5d'),
        'sequence_logit_fn':
        SequenceLogitsParams(
            use_attention=True,
            use_autoregression=True,
            num_lstm_units=256,
            weight_decay=0.00004,
            lstm_state_clip_value=10.0),
        'sequence_loss_fn':
        SequenceLossParams(
            label_smoothing=0,
            ignore_nulls=False,
            average_across_timesteps=False),
        'encode_coordinates_fn': EncodeCoordinatesParams(enabled=False)
    }
