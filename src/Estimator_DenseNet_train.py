import tensorflow as tf
import os
from densenet_keras import DenseNet


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
test_path = os.path.join('/data/TFRecord', 'test.tfrecords')


def create_model():
    dense_net = DenseNet(7, 12, 3, 10, 5,
                         bottleneck=True, compression=0.5, weight_decay=1e-4, dropout_rate=0.2, pool_initial=False,
                         include_top=True)

    return dense_net.build(input_shape=(128, 47, 2))


def model_fn(features, labels, mode, params):
    model = create_model()

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(0.001, momentum=0.9, use_nesterov=True)

        logits = model(features, training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) + tf.add_n(model.losses)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1, name='acc_op')
        )
        tf.summary.scalar('acc', accuracy[1])
        tf.summary.scalar('loss', loss)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step())
        )
    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(features, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1, name='acc_op')
        )
        tf.summary.scalar('eval_oss', loss)
        tf.summary.scalar('eval_acc', accuracy[1])
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={'accuracy': accuracy})

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(features, training=False)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            }
        )


def train_input_fn(args):
    """ like generator"""
    train_ds = tf.data.TFRecordDataset(train_path).map(parse_example).cache().shuffle(62000).apply(
        tf.contrib.data.batch_and_drop_remainder(args.batch_size)).repeat(args.epochs)
    return train_ds.make_one_shot_iterator().get_next()


def eval_input_fn(args):
    """ like generator"""
    return tf.data.TFRecordDataset(train_path).map(parse_example).batch(args.batch_size) \
        .make_one_shot_iterator().get_next()


def run_estimator_train(args):
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.output_dir,
        params={
            'data_format': args.data_format,
        })
    classifier.train(input_fn=lambda: train_input_fn(args), steps=61220 // args.batch_size, )
