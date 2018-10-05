import sklearn as sk
import pandas as pd
from tqdm import *
import librosa
import keras
import tensorflow as tf
import concurrent.futures
import os
import numpy as np
import datetime

# DATA_PATH = '/home/ccyoung/DCase/development_data/'

label_encoder = sk.preprocessing.LabelEncoder()
writer = None


def read_fold(filename):
    return pd.read_csv(os.path.join('../../evaluation', filename), sep='\t',
                       names=['file', 'scene'],  # 數組形式
                       converters={'file': lambda s: s.replace('audio/', '')})


train = read_fold('fold1_train.txt')
# self.validation_df = self.read_fold('fold1_validation.txt')  ## 约训练集的三分之一做验证集
test = read_fold('fold1_evaluate.txt')
label_encoder.fit(train['scene'])
n_scenes = len(label_encoder.classes_)


def extract_feature1(path):
    """
    使用hpss源分离并抽取mel特征 不足补0
    :return:
    """
    audio, sr = librosa.core.load(path, sr=48000, duration=10.0)  # mono

    y_harmonic, y_percussive = librosa.effects.hpss(audio)
    mel_harmonic = librosa.feature.melspectrogram(y_harmonic, sr=sr, hop_length=1024, fmax=24000)
    mel_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr, hop_length=1024, fmax=24000)
    mel_harmonic = librosa.power_to_db(mel_harmonic)
    mel_percussive = librosa.power_to_db(mel_percussive)

    mel_harmonic = (mel_harmonic - np.mean(mel_harmonic)) / np.std(mel_harmonic)
    mel_percussive = (mel_percussive - np.mean(mel_percussive)) / np.std(mel_percussive)

    padding_data = np.zeros((mel_harmonic.shape[0], 1))  # 补充一列，以便均等分片
    mel_spec = np.stack([np.concatenate([mel_harmonic, padding_data], axis=1),
                         np.concatenate([mel_percussive, padding_data], axis=1)], axis=2)
    return mel_spec


def extract_feature2(path):
    """
    使用hpss源分离并抽取mel特征,不足补最后一列
    :return:
    """
    audio, sr = librosa.core.load(path, sr=48000, duration=10.0)  # mono

    y_harmonic, y_percussive = librosa.effects.hpss(audio)
    mel_harmonic = librosa.feature.melspectrogram(y_harmonic, sr=sr, hop_length=1024, fmax=24000)
    mel_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr, hop_length=1024, fmax=24000)
    mel_harmonic = librosa.power_to_db(mel_harmonic)
    mel_percussive = librosa.power_to_db(mel_percussive)

    mel_harmonic = (mel_harmonic - np.mean(mel_harmonic)) / np.std(mel_harmonic)
    mel_percussive = (mel_percussive - np.mean(mel_percussive)) / np.std(mel_percussive)

    mel_harmonic = np.concatenate([mel_harmonic, mel_harmonic[:, -1][..., None]], axis=1)
    mel_percussive = np.concatenate([mel_percussive, mel_percussive[:, -1][..., None]], axis=1)
    mel_spec = np.stack([mel_harmonic, mel_percussive], axis=2)

    return mel_spec


def process1(writer, file, scene):
    # path = os.path.join('/home/ccyoung/Downloads/2018_task1_A/TUT-urban-acoustic-scenes-2018-development-data/',file)
    path = os.path.join('/data/TUT-urban-acoustic-scenes-2018-development-data/', file)
    # 必须先转成float16，否则librosa无法处理

    mel_spec = extract_feature2(path)
    # print('mel_spec shape:', mel_spec.shape)

    # 开始分片 non-overlap 1s 47帧
    start = 0
    win_len = 47
    label = label_encoder.transform([scene])[0]
    label = keras.utils.to_categorical(label, 10)
    while start < mel_spec.shape[1]:
        end = start + win_len
        patch = mel_spec[:, start:end, :]
        patch = patch.reshape(-1)

        example = encapsulate_example(patch, label)
        writer.write(example.SerializeToString())

        start = end


def generate_non_overlap_TFRecord(dataset, tfrecord_path):
    """
    use with extract_feature1
    :param dataset:
    :param tfrecord_path:
    :return:
    """

    scenes = [row.scene for row in dataset.itertuples()]
    files = [row.file for row in dataset.itertuples()]

    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # executor.map(process, files, scenes) // map无法传入非迭代器参数 所以使用submit
            for file, scene in tqdm(zip(files, scenes), total=len(files)):
                executor.submit(process1, writer, file, scene)
    except Exception as err:
        print(err)
        writer.close()
    writer.close()


def process2(writer, file, scene):
    path = os.path.join('/data/TUT-urban-acoustic-scenes-2018-development-data/', file)
    # path = os.path.join('/home/ccyoung/Downloads/2018_task1_A/TUT-urban-acoustic-scenes-2018-development-data/',file)
    # 必须先转成float16，否则librosa无法处理

    mel_spec = extract_feature2(path)
    # print('mel_spec shape:', mel_spec.shape)

    # 开始分片 overlap1s 2s 94帧
    start = 0
    win_len = 94
    overlap = 47
    label = label_encoder.transform([scene])[0]
    label = keras.utils.to_categorical(label, 10)
    end = start + win_len
    while end <= mel_spec.shape[1]:
        patch = mel_spec[:, start:end, :]
        patch = patch.reshape(-1)

        example = encapsulate_example(patch, label)
        writer.write(example.SerializeToString())

        start += overlap
        end += overlap


def generate_overlap_TFRecord(dataset, tfrecord_path):
    """
    use with extract_feature2
    :param dataset:
    :param tfrecord_path:
    :return:
    """
    scenes = [row.scene for row in dataset.itertuples()]
    files = [row.file for row in dataset.itertuples()]

    with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # executor.map(process, files, scenes) // map无法传入非迭代器参数 所以使用submit
            for file, scene in tqdm(zip(files, scenes), total=len(files)):
                executor.submit(process2, writer, file, scene)


def encapsulate_example(audio, label):
    """
    封装样本
    :param audio:
    :param label:
    :return:
    """
    feature = {
        'audio': tf.train.Feature(
            float_list=tf.train.FloatList(value=audio)),
        'label': tf.train.Feature(
            float_list=tf.train.FloatList(value=label))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


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


def generate_tfrecords():
    """
    训练集重叠 测试集不重叠
    :return:
    """
    # path_prefix = '/home/ccyoung/DCase/Task1_2018/evaluation'
    path_prefix = '/data/TFRecord'
    path = os.path.join(path_prefix, 'train2.tfrecords')
    test_path = os.path.join(path_prefix, 'test2.tfrecords')
    # generate_overlap_TFRecord(train, path)
    generate_non_overlap_TFRecord(test, test_path)
    # os.system('sh /data/stop.sh')


def compute_time_consumed(start_time):
    """
    计算训练总耗时
    :param start_time:
    :return:
    """
    time_elapsed = datetime.datetime.now() - start_time
    seconds = time_elapsed.seconds
    hour = seconds // 3600
    minute = (seconds % 3600) // 60
    second = seconds % 3600 % 60
    print("本次训练共耗时 {0} 时 {1} 分 {2} 秒".format(hour, minute, second))


def main():
    # path_prefix = '/home/ccyoung/DCase/Task1_2018/evaluation'
    # generate_non_overlap_TFRecord(train, os.path.join(path_prefix, 'train2.tfrecords'))
    tf.enable_eager_execution()
    # path_prefix = '/home/ccyoung/DCase/Task1_2018/evaluation'
    # dataset = tf.data.TFRecordDataset(os.path.join(path_prefix, 'train2.tfrecords'))
    # dataset = dataset.map(parse_example)
    # cnt = 0
    # for a in dataset:
    #     cnt += 1
    #     print(cnt)
    #     print(a[0].shape)
    #     print(a[1].shape)
    # task.extract_feature2('../airport-barcelona-0-0-a.wav')

    # start_time = datetime.now()
    # generate_overlap_tfrecords()
    # compute_time_consumed(start_time)
    # generate_tfrecords()
    path_prefix = '/data/TFRecord'
    test_path = os.path.join(path_prefix, 'test2.tfrecords')
    dataset = tf.data.TFRecordDataset(test_path)
    dataset = dataset.map(parse_example).
    cnt = 0
    try:
        for _ in dataset:
            cnt += 1
        print(cnt)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
