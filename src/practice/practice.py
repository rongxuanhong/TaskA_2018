import librosa
import numpy as np
import tensorflow as tf


def generate_spectrum():
    y, sr = librosa.load('../airport-barcelona-0-0-a.wav', sr=48000, duration=10.0)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    mel_harmonic = librosa.power_to_db(
        librosa.feature.melspectrogram(y_harmonic, sr=sr, n_fft=4096, hop_length=3072, n_mels=128, fmax=24000))
    # mel_percussive = librosa.power_to_db(
    #     librosa.feature.melspectrogram(y_percussive, sr=sr, hop_length=1024, fmax=24000))
    mel_harmonic = np.concatenate((mel_harmonic, mel_harmonic[:, -11:]), axis=-1)

    print(mel_harmonic.shape)
    mel_harmonic = (mel_harmonic - np.mean(mel_harmonic)) / np.std(mel_harmonic)
    # mel_percussive = (mel_percussive - np.mean(mel_percussive)) / np.std(mel_percussive)
    # print(mel_harmonic[0, :10])
    # writer = tf.python_io.TFRecordWriter(path='train.tfrecords')
    # example = tf.train.Example(
    #     features=tf.train.Features(
    #         feature={
    #             "patch": tf.train.Feature(float_list=tf.train.FloatList(value=[mel_harmonic.reshape(-1)])),
    #             "label": tf.train.Feature(float_list=tf.train.FloatList(value=[mel_percussive.reshape(-1)]))
    #         }
    #     )
    # )
    #
    # writer.write(example.SerializeToString())

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
    # learning_rates = [0.01, 0.01 * 0.1, 0.01 * 0.01]
    # boundaries = [int(0.4 * 30), int(0.75 * 30)]
    # check_point = tf.train.Checkpoint(a='')
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     for global_step in range(30):
    #         learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=learning_rates)
    #         a = sess.run([learning_rate])
    #         print(a)
    step_counter = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(0.01, step_counter, decay_steps=10, decay_rate=0.96)
    # train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=current_epoch)...
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(40):
            lr = sess.run([learning_rate])
            print(lr)


def main():
    # coding:utf-8
    import matplotlib.pyplot as plt
    import tensorflow as tf

    num_epoch = tf.Variable(0, name='global_step', trainable=False)

    y = []
    z = []
    N = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for num_epoch in range(N):
            # 阶梯型衰减
            learing_rate1 = tf.train.exponential_decay(
                learning_rate=0.001, global_step=num_epoch, decay_steps=1, decay_rate=0.9, staircase=True)
            # 标准指数型衰减
            learing_rate2 = tf.train.exponential_decay(
                learning_rate=0.001, global_step=num_epoch, decay_steps=2, decay_rate=0.5, staircase=False)

            lr1 = sess.run([learing_rate1])
            lr2 = sess.run([learing_rate2])
            print(lr2)
            y.append(lr1)
            z.append(lr2)

    x = range(N)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_ylim([0, 0.55])

    plt.plot(x, y, 'r-', linewidth=2)
    plt.plot(x, z, 'g-', linewidth=2)
    plt.title('exponential_decay')
    ax.set_xlabel('step')
    ax.set_ylabel('learing rate')
    plt.show()


if __name__ == '__main__':
    # main()
    # generate_spectrum()
    # import os
    #
    # os.system('sh /data/stop_instance.sh')
    boundaries = []
    learning_rate = 0.001
    learning_rates = [learning_rate]
    decay_rate = 0.5
    for i in range(9):
        if (i + 1) % 2 == 0:
            boundaries.append(2)
            learning_rate *= decay_rate
            learning_rates.append(learning_rate)
    for i in boundaries:
        print(i)
    for i in learning_rates:
        print(i)
