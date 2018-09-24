from sklearn import metrics
import numpy as np
import os
from datetime import datetime


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
    os.system('sh /data/stop_instance.sh')


def mix_data(x, y, alpha=1.0):
    """ mixup data augmentation"""
    lam = np.random.beta(alpha, alpha)

    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b,lam


