import numpy as np
import sklearn as sk
import pandas as pd
from tqdm import *
import librosa
import os
import keras
# import config as cfg
from file import project_dir
import tensorflow as tf
from utils import *

DATA_PATH = '/home/ccyoung/DCase/development_data/'


class DateSet:
    def __init__(self, args):
        self.train = None
        self.test = None
        self.label_encoder = sk.preprocessing.LabelEncoder()
        self.n_scenes = None
        self.args = args
        self.slice_len = 47
        self.batch_size = 2

    def read_fold(self, filename):
        return pd.read_csv(os.path.join('../../evaluation', filename), sep='\t',
                           names=['file', 'scene'],  # 數組形式
                           converters={'file': lambda s: s.replace('audio/', '')})

    def load_dataset(self):

        self.train = self.read_fold('fold1_train.txt')
        # self.validation_df = self.read_fold('fold1_validation.txt')  ## 约训练集的三分之一做验证集
        self.test = self.read_fold('fold1_evaluate.txt')
        self.label_encoder.fit(self.train['scene'])
        self.n_scenes = len(self.label_encoder.classes_)

    def extract_feature(self, path):
        """
        使用hpss源分离并抽取mel特征
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

    def generate_TFRecord(self, dataset, tfrecord_path):
        """

        :param dataset:
        :param tfrecord_path:
        :return:
        """

        writer = tf.python_io.TFRecordWriter(path=tfrecord_path)

        for row in tqdm(dataset.itertuples(), total=len(dataset)):
            path = os.path.join('/data/TUT-urban-acoustic-scenes-2018-development-data/', row.file)
            # 必须先转成float16，否则librosa无法处理

            mel_spec = self.extract_feature(path)
            # print('mel_spec shape:', mel_spec.shape)

            # 开始分片 non-overlap 1s 47帧
            start = 0
            win_len = 47
            label = self.label_encoder.transform([row.scene])[0]
            label = keras.utils.to_categorical(label, self.n_scenes)
            while start < mel_spec.shape[1]:
                end = start + win_len
                patch = mel_spec[:, start:end, :]
                patch = patch.reshape(-1)

                example = self.encapsulate_example(patch, label)
                writer.write(example.SerializeToString())

                start = end
        writer.close()

    def encapsulate_example(self, audio, label):
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

    def parse_example(self, example):
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


def main():
    path_prefix = '/data/TFRecord'
    path = os.path.join(path_prefix, 'train.tfrecords')
    test_path = os.path.join(path_prefix, 'test.tfrecords')
    task = DateSet(None)
    task.load_dataset()
    task.generate_TFRecord(task.train, path)
    os.system('sh /data/stop_instance.sh')
    # task.generate_TFRecord(task.test, test_path)
    # dataset = tf.data.TFRecordDataset(path)
    # dataset = dataset.map(task.parse_example)

    # iterator = dataset.make_initializable_iterator()
    # next_element = iterator.get_next()

    # sess = tf.Session()
    count = 0
    # data=list()
    # for _ in range(100):
    #     sess.run(iterator.initializer)
    #     while True:
    #         try:
    #             a = sess.run(next_element)
    #             print(a)
    #         except tf.errors.OutOfRangeError:
    #             brea


if __name__ == '__main__':
    main()
