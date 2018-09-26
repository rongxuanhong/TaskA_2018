from denset_net import DenseNet
import tensorflow.contrib as tfc
import argparse
from utils.utils import *
from datetime import datetime
import os

"""使用 Eager Execution编写， 适合与 NumPy 一起使用"""


def loss(logits, labels):
    """
    定义损失函数
    :param logits:
    :param labels:
    :return:
    """
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels))


def compute_accuracy(logits, labels):
    """
    计算准确度
    :param logits:
    :param labels:
    :return:
    """
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    labels = tf.argmax(labels, axis=1, output_type=tf.int64)  ##labels采用的是独热编码的，也要tf.argmax
    return tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.float32)) / int(logits.shape[0]) * 100


def compute_mix_accuracy(logits, label_a, label_b, lam):
    """
    计算准确度
    :param logits:
    :param labels:
    :return:
    """
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    label_a = tf.argmax(label_a, axis=1, output_type=tf.int64)
    label_b = tf.argmax(label_b, axis=1, output_type=tf.int64)
    correct = lam * tf.reduce_sum(tf.cast(tf.equal(predictions, label_a), tf.float32)) + \
              (1 - lam) * tf.reduce_sum(tf.cast(tf.equal(predictions, label_b), tf.float32))
    return correct / int(logits.shape[0]) * 100


def train(model, optimizer, dataset, step_counter, total_batch, log_interval, args):
    """
    在dataset上使用optimizer训练model，
    :param model:
    :param optimizer:
    :param dataset:
    :param step_counter:
    :param total_batch:每轮总批次
    :return:
    """
    for batch, (audios, labels) in enumerate(dataset):  # 遍历一次数据集
        with tfc.summary.record_summaries_every_n_global_steps(
                10, global_step=step_counter):
            with tf.GradientTape() as tape:
                audios = tf.reshape(audios, (args.batch_size, 128, 47, 2))
                # mixed_audios, label_a, label_b, lam = mix_data(audios, labels, args.batch_size, args.alpha)
                logits = model(audios, training=True)

                # 计算损失
                # loss_value = lam * loss(logits, label_a) + (1 - lam) * loss(logits, label_b)

                loss_value = loss(logits, labels)
                # 每10步记录日志
                # acc = compute_mix_accuracy(logits, label_a, label_b, lam)
                acc = compute_accuracy(logits, labels)

                tfc.summary.scalar('loss', loss_value)
                tfc.summary.scalar('accuracy', acc)
            # 梯度求解
            grads = tape.gradient(loss_value, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables), global_step=step_counter)

        # 打印log
        if log_interval and batch % log_interval == 0:
            print('Step：{0:2d}/{1}  loss:{2:.6f} acc:{3:.2f}'.format(batch, total_batch, loss_value,
                                                                     compute_accuracy(logits, labels)))


def test(model, dataset, args):
    """
    使用評估集測試模型效果(带答案的评估集或称测试集)
    :param model:
    :param dataset: 测试集
    :return:
    """
    avg_loss = tfc.eager.metrics.Mean('loss', dtype=tf.float32)
    accuracy = tfc.eager.metrics.Accuracy('accuracy', dtype=tf.float32)

    for (audios, labels) in dataset:
        audios = tf.reshape(audios, (args.batch_size, 128, 47, 2))
        logits = model(audios, training=False)
        avg_loss(loss(logits, labels))
        accuracy(
            tf.argmax(logits, axis=1, output_type=tf.int64),
            tf.argmax(labels, axis=1, output_type=tf.int64)
        )
    print('Test set: Average loss: %.4f, Accuracy: %2f%%\n' %
          (avg_loss.result(), 100 * accuracy.result()))
    with tfc.summary.always_record_summaries():
        tfc.summary.scalar('test_loss', avg_loss.result())
        tfc.summary.scalar('test_acc', accuracy.result())


def predict_evaluation():
    """
    预测评估集的标签
    :return:
    """
    pass


def verify_model(dataset, model):
    """
    验证集评估模型
    :param dataset:
    :param model:
    :return:
    """
    avg_loss = tfc.eager.metrics.Mean('loss', dtype=tf.float32)
    accuracy = tfc.eager.metrics.Accuracy('accuracy', dtype=tf.float32)

    for (audios, labels) in dataset:
        logits = model(audios, training=False)
        avg_loss(loss(logits, labels))
        accuracy(
            tf.argmax(logits, axis=1, output_type=tf.int64),
            tf.argmax(labels, axis=1, output_type=tf.int64)
        )
    print('validation set: Average loss: %.4f, Accuracy: %4f%%\n' %
          (avg_loss.result(), 100 * accuracy.result()))
    with tfc.summary.always_record_summaries():
        tfc.summary.scalar('val_loss', avg_loss.result())
        tfc.summary.scalar('val_acc', accuracy.result())


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


def run_task_eager(args):
    """
    在急切模式下，运行模型的训练以及评估
    :param args:包含parsed flag 值的对象
    :return:
    """
    # 1. 手动开始急切执行
    tf.enable_eager_execution()

    # dateset = DateSet(args)
    # dateset.load_dataset()
    # task.prepare_data()

    # 2.自动决定运行设备和数据格式
    # (device, data_format) = ('/gpu:0', 'channels_first')
    # if args.no_gpu or not tf.test.is_gpu_available():
    #     (device, data_format) = ('/cpu:0', 'channels_last')
    # If data_format is defined in FLAGS, overwrite automatically set value.
    # if args.data_format is not None:
    # data_format = args.data_format
    # print('Using device %s, and data format %s.' % (device, data_format))

    # 3.加载数据
    batch_size = args.batch_size
    total_batch = 61220 // batch_size
    # train_ds = tf.data.Dataset.from_tensor_slices(task.train).shuffle(10000).batch(
    #     args.batch_size)
    # test_ds = tf.data.Dataset.from_tensor_slices(task.test).batch(args.batch_size)
    train_path = os.path.join('/data/TFRecord', 'train.tfrecords')
    test_path = os.path.join('/data/TFRecord', 'test.tfrecords')
    # train_path = os.path.join('/home/ccyoung/DCase', 'train.tfrecords')
    # test_path = os.path.join('/home/ccyoung/DCase', 'test.tfrecords')
    train_ds = tf.data.TFRecordDataset(train_path).map(parse_example).shuffle(62000).apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
    test_ds = tf.data.TFRecordDataset(test_path).map(parse_example).apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    # 4.创建模型和优化器
    denset = DenseNet(input_shape=(128, 47, 2), n_classes=10, nb_layers=args.nb_layers,
                      nb_dense_block=args.n_db,
                      growth_rate=args.grow_rate)
    model = denset.build()
    describe_model(model)

    step_counter = tf.train.get_or_create_global_step()
    learning_rate = tf.train.piecewise_constant(step_counter, [int(0.5 * args.epochs), int(0.75 * args.epochs)],
                                                [args.lr, args.lr * 0.1, args.lr * 0.01])

    optimizer = tf.train.AdamOptimizer()
    # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)

    # 5. 创建用于写入tensorboard总结的文件写入器
    if args.output_dir:
        train_dir = os.path.join(args.output_dir, 'train')
        test_dir = os.path.join(args.output_dir, 'test')
        tf.gfile.MakeDirs(args.output_dir)  # 创建所有文件
    else:
        train_dir = None
        test_dir = None
    summary_writer = tfc.summary.create_file_writer(train_dir, flush_millis=10000, name='train')
    test_summary_writer = tfc.summary.create_file_writer(test_dir, flush_millis=10000, name='test')

    # 6. 创建或者恢复checkpoint
    check_point_prefix = os.path.join(args.output_dir, 'cpkt')
    create_folder(check_point_prefix)

    check_point = tf.train.Checkpoint(model=model, optimizer=optimizer, step_counter=step_counter)

    # check_point.restore(tf.train.latest_checkpoint(args.model_dir))  # 存在就恢复模型(可不使用)
    # 7. 训练、评估
    # with tf.device(device):
    start_time = datetime.now()
    for i in range(args.epochs):  # 迭代的轮次
        with summary_writer.as_default():
            # 训练
            print('epochs:{0}/{1}'.format((i + 1), args.epochs))
            train(model, optimizer, train_ds, step_counter, total_batch, args.log_interval, args)
            # 验证
            # verify_model(validation_ds, model)
        with test_summary_writer.as_default():
            # 测试
            # if (i + 1) % 5 == 0:
            # 评估
            test(model, test_ds, args)
        check_point.save(check_point_prefix)  # 保存检查点
    # 输出训练时间
    compute_time_consumed(start_time)


def define_task_eager_flags():
    """
    定义一些flags方便在终端运行项目
    :return:
    """
    arg = argparse.ArgumentParser()
    arg.add_argument('--batch_size', type=int, default=32)
    arg.add_argument('--epochs', type=int, default=40)
    arg.add_argument('--nb_layers', type=int, default=5)
    arg.add_argument('--n_db', type=int, default=3)
    arg.add_argument('--grow_rate', type=int, default=12)
    arg.add_argument('--data_format', type=str, required=True)
    arg.add_argument('--output_dir', type=str, required=True)
    arg.add_argument('--lr', type=float, required=True, default=0.001)
    arg.add_argument('--log_interval', type=int, required=True, default=10)
    arg.add_argument('--alpha', type=float, default=0.2)

    args = arg.parse_args()
    return args


def main(args):
    try:
        run_task_eager(args)
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
