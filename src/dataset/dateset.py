import sklearn as sk
import pandas as pd
from tqdm import *
import librosa
import keras
import tensorflow as tf
import os
import numpy as np
from datetime import datetime


# DATA_PATH = '/home/ccyoung/DCase/development_data/'


class DataSet:
    def __init__(self):
        self.train = None
        self.test = None
        self.label_encoder = sk.preprocessing.LabelEncoder()
        self.n_scenes = 10

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

    def extract_feature1(self, path):
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

    def extract_feature2(self, path):
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

    def extract_feature3(self, path):
        """
        使用hpss源分离并抽取mel特征 不足补0
        :return:
        """
        audio, sr = librosa.core.load(path, sr=44100, duration=10.0)  # mono

        y_harmonic, y_percussive = librosa.effects.hpss(audio)

        mel_harmonic = librosa.feature.melspectrogram(y_harmonic, sr=sr, n_fft=2205, hop_length=882, n_mels=100)
        mel_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr, n_fft=2205, hop_length=882, n_mels=100)

        mel_harmonic = librosa.power_to_db(mel_harmonic)
        mel_percussive = librosa.power_to_db(mel_percussive)

        mel_harmonic = (mel_harmonic - np.mean(mel_harmonic)) / np.std(mel_harmonic)
        mel_percussive = (mel_percussive - np.mean(mel_percussive)) / np.std(mel_percussive)

        mel_spec = np.stack([mel_harmonic, mel_percussive], axis=-1)
        return mel_spec

    def extract_feature4(self, path):
        """
        使用hpss源分离并抽取mel特征,不足补最后一列
        :return:
        """
        audio, sr = librosa.core.load(path, sr=48000, duration=10.0)  # mono

        y_harmonic, y_percussive = librosa.effects.hpss(audio)
        mel_harmonic = librosa.feature.melspectrogram(y_harmonic, sr=sr, n_fft=1920, hop_length=960, n_mels=64,
                                                      fmax=24000)
        mel_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr, n_fft=1920, hop_length=960, n_mels=64,
                                                        fmax=24000)
        mel_harmonic = librosa.power_to_db(mel_harmonic)
        mel_percussive = librosa.power_to_db(mel_percussive)

        mel_harmonic = (mel_harmonic - np.mean(mel_harmonic)) / np.std(mel_harmonic)
        mel_percussive = (mel_percussive - np.mean(mel_percussive)) / np.std(mel_percussive)

        mel_harmonic = np.concatenate([mel_harmonic, mel_harmonic[:, -11:]], axis=-1)
        mel_percussive = np.concatenate([mel_percussive, mel_percussive[:, -11:]], axis=-1)
        mel_spec = np.stack([mel_harmonic, mel_percussive], axis=-1)

        return mel_spec

    def extract_feature5(self, path):
        """
        :return:
        """
        audio, sr = librosa.core.load(path, sr=48000, duration=10.0)  # mono

        mel = librosa.feature.melspectrogram(audio, sr=sr, n_fft=4096, hop_length=3072, n_mels=128,
                                             fmax=24000)
        mel = librosa.power_to_db(mel)

        mel = (mel - np.mean(mel)) / np.std(mel)

        return mel[..., None]

    def extract_feature6(self, path):
        """
        :return:
        """
        audio, sr = librosa.core.load(path, sr=48000, duration=10.0)  # mono

        audio = random_shifting(audio)
        mel = librosa.feature.melspectrogram(audio, sr=sr, n_fft=4096, hop_length=3072, n_mels=128,
                                             fmax=24000)
        mel = librosa.power_to_db(mel)

        mel = (mel - np.mean(mel)) / np.std(mel)

        return mel[..., None]

    def extract_feature7(self, path):
        """
        :return:
        """
        audio, sr = librosa.core.load(path, sr=48000, duration=10.0)  # mono

        audio = add_gaussian_noise(audio)
        mel = librosa.feature.melspectrogram(audio, sr=sr, n_fft=4096, hop_length=3072, n_mels=128,
                                             fmax=24000)
        mel = librosa.power_to_db(mel)

        mel = (mel - np.mean(mel)) / np.std(mel)

        return mel[..., None]

    def extract_feature8(self, path, augment=True):
        """
        :return:
        """
        y, sr = librosa.core.load(path, sr=48000, duration=10.0, mono=False)  # mono

        y_list = [y]
        if augment:
            for pitch in [-2.0]:
                y1 = librosa.effects.pitch_shift(y[0], sr, n_steps=pitch, )
                y2 = librosa.effects.pitch_shift(y[1], sr, n_steps=pitch, )
                y_list.append(np.stack([y1, y2]))

        mel_list = []

        for y in y_list:
            power = np.abs(librosa.stft(y[0], n_fft=4096, hop_length=3072)) ** 2  # 1025,431
            power = librosa.core.perceptual_weighting(power, frequencies=librosa.fft_frequencies(sr=48000, n_fft=4096))
            mel = librosa.feature.melspectrogram(S=power, n_mels=128)
            mel1 = (mel - np.mean(mel)) / np.std(mel)

            power = np.abs(librosa.stft(y[1], n_fft=4096, hop_length=3072)) ** 2  # 1025,431
            power = librosa.core.perceptual_weighting(power, frequencies=librosa.fft_frequencies(sr=48000, n_fft=4096))
            mel = librosa.feature.melspectrogram(S=power, n_mels=128)
            mel2 = (mel - np.mean(mel)) / np.std(mel)

            # power = np.abs(librosa.stft(y[0] - y[1], n_fft=2048, hop_length=512)) ** 2  # 1025,431
            # power = librosa.core.perceptual_weighting(power, frequencies=librosa.fft_frequencies())
            # mel = librosa.feature.melspectrogram(S=power, n_mels=256)
            # mel3 = (mel - np.mean(mel)) / np.std(mel)

            # indexs=np.random.choice(mel1.shape[1],128)
            mel = np.stack([mel1, mel2], axis=-1)
            # print(mel.shape)
            mel_list.append(mel)

        return mel_list

    def generate_TFRecord(self, dataset, tfrecord_path, augment=True):
        """
        use with extract_feature1
        :param dataset:
        :param tfrecord_path:
        :return:
        """

        writer = tf.python_io.TFRecordWriter(tfrecord_path)
        for row in tqdm(dataset.head(5).itertuples(), total=len(dataset)):
            path = os.path.join('/home/ccyoung/Downloads/2018_task1_A/TUT-urban-acoustic-scenes-2018-development-data/',
                                row.file)
            # path = os.path.join('/data/TUT-urban-acoustic-scenes-2018-development-data/', row.file)
            # 必须先转成float16，否则librosa无法处理

            label = self.label_encoder.transform([row.scene])[0]
            label = keras.utils.to_categorical(label, self.n_scenes)

            mel_spec_list = self.extract_feature8(path, augment)
            for mel_spec in mel_spec_list:
                example = self.encapsulate_example(mel_spec.reshape(-1), label)
                writer.write(example.SerializeToString())

        writer.close()

    def generate_non_overlap_TFRecord(self, dataset, tfrecord_path):
        """
        use with extract_feature1
        :param dataset:
        :param tfrecord_path:
        :return:
        """

        writer = tf.python_io.TFRecordWriter(tfrecord_path)
        for row in tqdm(dataset.itertuples(), total=len(dataset)):
            # path = os.path.join('/home/ccyoung/Downloads/2018_task1_A/TUT-urban-acoustic-scenes-2018-development-data/',
            #                     row.file)
            path = os.path.join('/data/TUT-urban-acoustic-scenes-2018-development-data/', row.file)
            # 必须先转成float16，否则librosa无法处理

            mel_spec = self.extract_feature4(path)
            # print('mel_spec shape:', mel_spec.shape)

            # 开始分片 non-overlap 1s 47帧
            start = 0
            win_len = 64
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

    def generate_overlap_TFRecord(self, dataset, tfrecord_path):
        """
        use with extract_feature2
        :param dataset:
        :param tfrecord_path:
        :return:
        """
        writer = tf.python_io.TFRecordWriter(tfrecord_path)
        for row in tqdm(dataset.itertuples(), total=len(dataset)):

            path = os.path.join('/data/TUT-urban-acoustic-scenes-2018-development-data/', row.file)
            # path = os.path.join('/home/ccyoung/Downloads/2018_task1_A/TUT-urban-acoustic-scenes-2018-development-data/',
            #                     row.file)
            # 必须先转成float16，否则librosa无法处理

            mel_spec = self.extract_feature4(path)

            # 开始分片 overlap1s 2s 94帧
            start = 0
            win_len = 64
            overlap = 32
            label = self.label_encoder.transform([row.scene])[0]
            label = keras.utils.to_categorical(label, self.n_scenes)
            end = start + win_len
            while end <= mel_spec.shape[1]:
                patch = mel_spec[:, start:end, :]
                patch = patch.reshape(-1)

                example = self.encapsulate_example(patch, label)
                writer.write(example.SerializeToString())

                start += overlap
                end += overlap
        writer.close()

    def generate_overlap_TFRecord2(self, dataset, tfrecord_path):
        """
        use with extract_feature2
        :param dataset:
        :param tfrecord_path:
        :return:
        """
        writer = tf.python_io.TFRecordWriter(tfrecord_path)
        for row in tqdm(dataset.itertuples(), total=len(dataset)):

            path = os.path.join('/data/TUT-urban-acoustic-scenes-2018-development-data/', row.file)
            # path = os.path.join('/home/ccyoung/Downloads/2018_task1_A/TUT-urban-acoustic-scenes-2018-development-data/',
            #                     row.file)
            # 必须先转成float16，否则librosa无法处理

            mel_spec = self.extract_feature3(path)
            # print('mel_spec shape:', mel_spec.shape)

            # 开始分片 overlap1s 2s 94帧
            start = 0
            win_len = 100
            overlap = 50
            label = self.label_encoder.transform([row.scene])[0]
            label = keras.utils.to_categorical(label, self.n_scenes)
            end = start + win_len
            while end <= mel_spec.shape[1]:
                patch = mel_spec[:, start:end, :]
                patch = patch.reshape(-1)

                example = self.encapsulate_example(patch, label)
                writer.write(example.SerializeToString())

                start += overlap
                end += overlap
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


def generate_small_data():
    """
    分层抽样
    :return:
    """
    path_prefix = '/data/TFRecord'
    task = DataSet()
    task.load_dataset()
    data = task.train
    result = data.groupby('scene', group_keys=False).apply(lambda x: x.sample(frac=0.2))
    print(len(result))
    task.generate_TFRecord(result, os.path.join(path_prefix, 'small_data.tfrecords'))


def random_shifting(y):
    """
    随机移动 onset
    :param y:
    :return:
    """
    timeshift_fac = 0.2 * 2 * (np.random.uniform() - 0.5)  # up to 20% of length
    start = int(y.shape[0] * timeshift_fac)
    if (start > 0):
        y_shift = np.pad(y, (start, 0), mode='constant')[0:y.shape[0]]  # 信号右移(数组前面填充start个0，再取前y.shape[0]个采样点)
    else:
        y_shift = np.pad(y, (0, -start), mode='constant')[0:y.shape[0]]  # 信号左移
    return y_shift


def add_gaussian_noise(y):
    """
    添加分布噪声,如高斯噪声
    :param y:
    :return:
    """
    noise_amap = 0.005 * np.random.uniform() * np.amax(y)
    y_noise = y + noise_amap.astype(np.float32) * np.random.normal(size=len(y))
    return y_noise


def main():
    path_prefix = '/home/ccyoung/DCase/Task1_2018/evaluation'
    # path_prefix = '/data/TFRecord'
    # tf.enable_eager_execution()

    start_time = datetime.now()

    task = DataSet()
    task.load_dataset()
    # task.generate_overlap_TFRecord(task.train, os.path.join(path_prefix, 'train4.tfrecords'))
    # task.generate_non_overlap_TFRecord(task.test, os.path.join(path_prefix, 'test4.tfrecords'))
    # os.system('sh /data/stop_instance.sh')
    # mel = task.extract_feature8('../airport-barcelona-0-0-a.wav')
    # print(mel.shape)
    task.generate_TFRecord(task.train, os.path.join(path_prefix, 'train11.tfrecords'))
    # task.generate_TFRecord(task.test, os.path.join(path_prefix, 'test11.tfrecords'), augment=False)
    compute_time_consumed(start_time)
    os.system('sh /data/stop.sh')


if __name__ == '__main__':
    main()
    # generate_small_data()
    # y, sr = librosa.load(librosa.util.example_audio_file())
    # CQT = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('A1'))
    # freqs = librosa.cqt_frequencies(CQT.shape[0], fmin=librosa.note_to_hz('A1'))
    # print(freqs)
    # perceptual_CQT = librosa.perceptual_weighting(CQT ** 2, freqs, ref=np.max)
    # print(perceptual_CQT)
