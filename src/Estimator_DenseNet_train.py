import tensorflow as tf
import os


def parse_example(example):
    """
    解析样本
    :param example:
    :return:
    """
    keys_to_features = {
        'audio': tf.VarLenFeature(tf.float32),
        'label': tf.VarLenFeature(tf.float32),
    }
    parsed = tf.parse_single_example(example, keys_to_features)
    audios = tf.sparse_tensor_to_dense(parsed['audio'], default_value=0)
    labels = tf.sparse_tensor_to_dense(parsed['label'], default_value=0)
    return audios, labels


train_path = os.path.join('/data/TFRecord', 'train.tfrecords')


def train_input_fn(batch_size):
    train_ds = tf.data.TFRecordDataset(train_path).map(parse_example).shuffle(62000).apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
    return train_ds.make_one_shot_iterator().get_next()
