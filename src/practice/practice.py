import librosa
import numpy as np
import tensorflow as tf


def generate_spectrum():
    y, sr = librosa.load('../airport-barcelona-0-0-a.wav', sr=48000, duration=10.0)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    mel_harmonic = librosa.power_to_db(librosa.feature.melspectrogram(y_harmonic, sr=sr, hop_length=1024))
    mel_percussive = librosa.power_to_db(librosa.feature.melspectrogram(y_percussive, sr=sr, hop_length=1024))

    mel_harmonic = (mel_harmonic - np.mean(mel_harmonic)) / np.std(mel_harmonic)
    mel_percussive = (mel_percussive - np.mean(mel_percussive)) / np.std(mel_percussive)
    # print(mel_harmonic[0, :10])
    writer = tf.python_io.TFRecordWriter(path='train.tfrecords')
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "patch": tf.train.Feature(float_list=tf.train.FloatList(value=[mel_harmonic.reshape(-1)])),
                "label": tf.train.Feature(float_list=tf.train.FloatList(value=[mel_percussive.reshape(-1)]))
            }
        )
    )

    writer.write(example.SerializeToString())

    # y = np.stack((mel_harmonic, mel_percussive), axis=2)
    # tf.data.TFRecordDataset(['a.tfrecord','b.tfrecord'])
    pass


def read_tfrecord():
    file_queue = tf.train.string_input_producer(['train.tfrecords'])
    reader = tf.TFRecordReader()
    _, example = reader.read(file_queue)
    features = tf.parse_single_example(example, features={
        'patch': tf.VarLenFeature(tf.float32),
        'label': tf.VarLenFeature(tf.float32),
    })
    map = tf.sparse_tensor_to_dense(features['patch'], default_value=0)
    map = tf.reshape(map, [128, 469])
    print(map)


def test_piece_constant():
    """
    测试学习率手动衰减
    :return:
    """
    learning_rates = [0.1, 0.01, 0.001]
    boundaries = [15, 23]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for global_step in range(30):
            learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=learning_rates)
            a = sess.run([learning_rate])
            print(a)


def main():
    # generate_spectrum()

    pass


if __name__ == '__main__':
    main()
