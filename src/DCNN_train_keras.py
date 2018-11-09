from VggStyle import VGGStyle
import argparse
from utils.utils import *
from datetime import datetime
import os
import tensorflow as tf
from keras import backend as K
from callbacks.monitor_callback import MonitorCallBack
from tensorflow.contrib.eager.python import tfe


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
    return tf.reshape(audios, (128, 157, 2)), labels


def loss(logits, labels):
    """
    定义损失函数
    :param logits:
    :param labels:
    :return:

    """
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels))


def mix_up_loss(y_true, y_pred):
    pass


def compute_accuracy(labels, logits):
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


def train_inputs(train_path, batch_size):
    dataset = tf.data.TFRecordDataset(train_path).map(parse_example).shuffle(12500).batch(batch_size)
    return dataset


def to_generator(dataset):
    count=0
    for audios, labels in tfe.Iterator(dataset):
        count+=1
        print(count)
        yield audios.numpy(), labels.numpy()


def test_inputs(test_path, batch_size):
    dataset = tf.data.TFRecordDataset(test_path).map(parse_example).batch(batch_size)
    return dataset


def run_task_eager(args):
    """
    在急切模式下，运行模型的训练以及评估
    :param args:包含parsed flag 值的对象
    :return:
    """
    # 1. 手动开始急切执行
    tf.enable_eager_execution()
    tf.set_random_seed(0)
    np.random.seed(0)

    start_time = datetime.now()

    # 3.加载数据
    batch_size = args.batch_size
    total_batch = 12244 // batch_size

    # if  args.local:
    train_path = os.path.join('/data/TFRecord', 'train11.tfrecords')
    test_path = os.path.join('/data/TFRecord', 'test11.tfrecords')

    # 4.创建模型和优化器
    model = VGGStyle(num_classes=10, weight_decay=0, initializer='he_uniform')

    dummy_x = tf.zeros((100, 128, 157, 2))
    model._set_inputs(dummy_x, training=True)
    print("Number of variables in the model :", len(model.variables))
    model.summary()

    model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy',
                  metrics=['accuracy', compute_accuracy])

    test_generator = to_generator(test_inputs(test_path, args.batch_size))

    model.fit_generator(generator=to_generator(train_inputs(train_path, args.batch_size)),
                        steps_per_epoch=total_batch,
                        callbacks=[MonitorCallBack(model, test_generator=test_generator, args=args)],
                        epochs=args.epochs, )

    # 输出训练时间
    compute_time_consumed(start_time)


def define_task_eager_flags():
    """
    定义一些flags方便在终端运行项目
    :return:
    """
    arg = argparse.ArgumentParser()
    arg.add_argument('--batch_size', type=int, default=16)
    arg.add_argument('--epochs', type=int, default=5)
    arg.add_argument('--output_dir', type=str, default='/data')
    arg.add_argument('--lr', type=float, default=0.001)
    arg.add_argument('--log_interval', type=int, default=10)
    arg.add_argument('--alpha', type=float, default=0.2)

    args = arg.parse_args()
    return args


def main(args):
    # try:
    #     run_task_eager(args)
    #     finish_instance()
    # except:
    #     finish_instance()
    run_task_eager(args)
    # finish_instance()


def finish_instance():
    os.system('sh /data/stop.sh')


if __name__ == '__main__':
    args = define_task_eager_flags()
    main(args)
