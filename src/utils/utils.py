from sklearn import metrics
import numpy as np
import os
from datetime import datetime
import tensorflow as tf


def calculate_auc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred, average='macro')


def calculate_ap(y_true, y_pred):
    return metrics.average_precision_score(y_true, y_pred, average='macro')


def calculate_accuracy(y_true, y_pred):
    threshold_predict = (np.sign(y_pred - 0.5) + 1) / 2
    return np.sum(threshold_predict == y_true) / len(y_true)


def standard_scale(x, mean, std):
    return (x - mean) / std


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def describe_model(model):
    """
    描述keras模型的结构
    :param model:keras model
    :return:
    """

    description = 'Model layers / shapes / parameters:\n'
    total_params = 0

    for layer in model.layers:
        layer_params = layer.count_params()
        description += '- {}'.format(layer.name).ljust(20)
        description += '{}'.format(layer.input_shape).ljust(20)
        description += '{0:,}'.format(layer_params).rjust(12)
        description += '\n'
        total_params += layer_params

    description += 'Total:'.ljust(30)
    description += '{0:,}'.format(total_params).rjust(22)

    print(description)


def compute_time_consumed(start_time):
    """
    计算训练总耗时
    :param start_time:
    :return:
    """
    time_elapsed = datetime.now() - start_time
    seconds = time_elapsed.seconds
    hour = seconds // 3600
    minute = (seconds % 3600) // 60
    second = seconds % 3600 % 60
    print("本次训练共耗时 {0} 时 {1} 分 {2} 秒".format(hour, minute, second))


def finish_instance():
    import os
    os.system('sh /data/stop.sh')


def mix_data(x, y, batch_size, alpha=1.0):
    """ mixup data augmentation"""
    lam = np.random.beta(alpha, alpha)
    x = x.numpy()
    y = y.numpy()

    index = np.random.permutation(batch_size)

    mixed_x = tf.convert_to_tensor(lam * x + (1 - lam) * x[index, ...])
    y_a, y_b = tf.convert_to_tensor(y), tf.convert_to_tensor(y[index, :])
    return mixed_x, y_a, y_b, lam


def mix_data_generator(x, y, batch_size, alpha=0.2):
    """ mixup data augmentation"""
    lam = np.random.beta(alpha, alpha, batch_size)
    x = x.numpy()
    y = y.numpy()
    epochs = int(x) // batch_size
    for _ in epochs:
        index = np.random.permutation(batch_size)

        mixed_x = tf.convert_to_tensor(lam * x + (1 - lam) * x[index, ...])
        mixed_y = tf.convert_to_tensor(lam * y + (1 - lam) * y[index, ...])
        yield mixed_x, mixed_y
