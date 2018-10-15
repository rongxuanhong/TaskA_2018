import tensorflow as tf
import os
import argparse


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


train_path = os.path.join('/home/ccyoung/DCase', 'train.tfrecords')
test_path = os.path.join('/home/ccyoung/DCase', 'test.tfrecords')


def create_model():
    from densenet2 import DenseNet
    # dense_net = DenseNet(7, 12, 3, 10, 5,
    #                      bottleneck=True, compression=0.5, weight_decay=1e-4, dropout_rate=0.2, pool_initial=False,
    #                      include_top=True)

    # return dense_net.build(input_shape=(128, 47, 2)
    # )
    return DenseNet(7, args.grow_rate, args.n_db, 10, args.nb_layers, data_format=args.data_format,
                    bottleneck=True, compression=0.5, weight_decay=1e-4, dropout_rate=0.2, pool_initial=False,
                    include_top=True)


def model_fn(features, labels, mode, params):
    model = create_model()

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(0.001, momentum=0.9, use_nesterov=True)

        logits = model(features)
        print(labels.shape)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(labels, axis=1, output_type=tf.int64), logits=logits) + tf.add_n(model.losses)
        accuracy = tf.metrics.accuracy(
            labels=tf.argmax(labels, axis=1, output_type=tf.int64), predictions=tf.argmax(logits, axis=1, name='acc_op')
        )
        tf.summary.scalar('acc', accuracy[1])
        tf.summary.scalar('loss', loss)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step())
        )
    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(features)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(labels, axis=1, output_type=tf.int64), logits=logits)
        accuracy = tf.metrics.accuracy(
            labels=tf.argmax(labels, axis=1, output_type=tf.int64), predictions=tf.argmax(logits, axis=1, name='acc_op')
        )
        tf.summary.scalar('eval_oss', loss)
        tf.summary.scalar('eval_acc', accuracy[1])
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={'accuracy': accuracy})

    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     logits = model(features)
    #     predictions = {
    #         'classes': tf.argmax(logits, axis=1),
    #         'probabilities': tf.nn.softmax(logits),
    #     }
    #     return tf.estimator.EstimatorSpec(
    #         mode=tf.estimator.ModeKeys.PREDICT,
    #         predictions=predictions,
    #         export_outputs={
    #             'classify': tf.estimator.export.PredictOutput(predictions)
    #         }
    #     )


def train_input_fn(args):
    """ like generator"""
    train_ds = tf.data.TFRecordDataset(train_path).map(parse_example).shuffle(62000).apply(
        tf.contrib.data.batch_and_drop_remainder(args.batch_size)).repeat(args.epochs)
    audios, labels = train_ds.make_one_shot_iterator().get_next()
    audios = tf.reshape(audios, (args.batch_size, 128, 47, 2))
    labels = tf.cast(labels,tf.int64)
    return audios, labels


def eval_input_fn(args):
    """ like generator"""
    dataset = tf.data.TFRecordDataset(test_path).map(parse_example).apply(
        tf.contrib.data.batch_and_drop_remainder(args.batch_size))

    audios, labels = dataset.make_one_shot_iterator().get_next()
    audios = tf.reshape(audios, (args.batch_size, 128, 47, 2))
    labels = tf.cast(labels, tf.int64)
    return audios, labels


def run_estimator_train(args):
    estimator = tf.estimator.Estimator(
        # config=tf.estimator.RunConfig(
        #     model_dir=args.output_dir, save_summary_steps=100, keep_checkpoint_max=5,),
        model_fn=model_fn,
        # model_dir=args.output_dir,
        params={
            'data_format': args.data_format,
        })
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(args),
                                        max_steps=(61220 // args.batch_size) * args.epochs)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(args))

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def define_task_eager_flags():
    """
    定义一些flags方便在终端运行项目
    :return:
    """
    arg = argparse.ArgumentParser()
    arg.add_argument('--batch_size', type=int, default=1)
    arg.add_argument('--epochs', type=int, default=5)
    arg.add_argument('--nb_layers', type=int, default=2)
    arg.add_argument('--n_db', type=int, default=2)
    arg.add_argument('--grow_rate', type=int, default=12)
    arg.add_argument('--data_format', type=str, default='channels_last')
    arg.add_argument('--output_dir', type=str, default='/home/ccyoung/')
    arg.add_argument('--lr', type=float, default=0.001)
    arg.add_argument('--log_interval', type=int, default=10)
    arg.add_argument('--alpha', type=float, default=0.2)

    args = arg.parse_args()
    return args


def main(args):
    run_estimator_train(args)
    try:
        run_estimator_train(args)
        finish_instance()
    except:
        finish_instance()
    # run_task_eager(args)
    # finish_instance()


def finish_instance():
    os.system('sh /data/stop_instance.sh')


if __name__ == '__main__':
    args = define_task_eager_flags()
    main(args)
