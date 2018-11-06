import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, \
    Dense, Dropout, MaxPool2D, GlobalAveragePooling2D, Concatenate, GaussianNoise

from tensorflow.keras.regularizers import l2


class ConvBlock1(tf.keras.Model):
    def __init__(self, initializer='he_uniform', weight_decay=1e-5):
        super(ConvBlock1, self).__init__()
        self.conv1 = Conv2D(kernel_size=5,
                            filters=42,
                            padding='same',
                            use_bias=False,
                            strides=2,
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(weight_decay))
        self.conv2 = Conv2D(kernel_size=3,
                            filters=42,
                            strides=1,
                            padding='same',
                            use_bias=False,
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(weight_decay))
        self.batchnorm1 = BatchNormalization(axis=-1)
        self.batchnorm2 = BatchNormalization(axis=-1)
        self.maxpool = MaxPool2D(pool_size=2,
                                 strides=2,
                                 padding='same')
        self.noise = GaussianNoise(1.00)

    def call(self, inputs, training=None, mask=None):
        output = self.conv1(inputs)
        output = tf.nn.relu(self.batchnorm1(output))

        output = self.conv2(output)
        output = tf.nn.relu(self.batchnorm2(output))

        output = self.maxpool(output)
        output = self.noise(output, training=training)
        return output


class ConvBlock2(tf.keras.Model):
    def __init__(self, initializer='he_uniform', weight_decay=1e-5):
        super(ConvBlock2, self).__init__()
        self.conv1 = Conv2D(kernel_size=3,
                            filters=84,
                            padding='same',
                            use_bias=False,
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(weight_decay))
        self.conv2 = Conv2D(kernel_size=3,
                            filters=84,
                            padding='same',
                            use_bias=False,
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(weight_decay))
        self.batchnorm1 = BatchNormalization(axis=-1)
        self.batchnorm2 = BatchNormalization(axis=-1)
        self.maxpool = MaxPool2D(pool_size=2,
                                 strides=2,
                                 padding='same')
        self.noise = GaussianNoise(0.75)

    def call(self, inputs, training=None, mask=None):
        output = self.conv1(inputs)
        output = tf.nn.relu(self.batchnorm1(output))

        output = self.conv2(output)
        output = tf.nn.relu(self.batchnorm2(output))

        output = self.maxpool(output)
        output = self.noise(output, training=training)
        return output


class ConvBlock3(tf.keras.Model):
    def __init__(self, initializer='he_uniform', weight_decay=1e-5):
        super(ConvBlock3, self).__init__()
        self.conv1 = Conv2D(kernel_size=3,
                            filters=168,
                            padding='same',
                            use_bias=False,
                            strides=1,
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(weight_decay))
        self.conv2 = Conv2D(kernel_size=3,
                            filters=168,
                            strides=1,
                            padding='same',
                            use_bias=False,
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(weight_decay))
        self.conv3 = Conv2D(kernel_size=3,
                            filters=168,
                            strides=1,
                            padding='same',
                            use_bias=False,
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(weight_decay))
        self.conv4 = Conv2D(kernel_size=3,
                            filters=168,
                            strides=1,
                            padding='same',
                            use_bias=False,
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(weight_decay))
        self.batchnorm1 = BatchNormalization(axis=-1)
        self.batchnorm2 = BatchNormalization(axis=-1)
        self.batchnorm3 = BatchNormalization(axis=-1)
        self.batchnorm4 = BatchNormalization(axis=-1)
        self.dropout1 = Dropout(0.3)
        self.dropout2 = Dropout(0.3)
        self.dropout3 = Dropout(0.3)
        self.maxpool = MaxPool2D(pool_size=2,
                                 strides=2,
                                 padding='same')
        self.noise = GaussianNoise(0.75)

    def call(self, inputs, training=None, mask=None):
        output = self.conv1(inputs)
        output = tf.nn.relu(self.batchnorm1(output))
        output = self.dropout1(output, training)

        output = self.conv2(output)
        output = tf.nn.relu(self.batchnorm2(output))
        output = self.dropout2(output, training)

        output = self.conv3(output)
        output = tf.nn.relu(self.batchnorm3(output))
        output = self.dropout3(output, training)

        output = self.conv4(output)
        output = tf.nn.relu(self.batchnorm4(output))

        output = self.maxpool(output)
        output = self.noise(output, training=training)
        return output


class VGGStyle(tf.keras.Model):
    def __init__(self, num_classes, initializer='he_uniform', weight_decay=1e-5):
        super(VGGStyle, self).__init__()
        self.convblock1 = ConvBlock1(initializer, weight_decay)
        self.convblock2 = ConvBlock2(initializer, weight_decay)
        self.convblock3 = ConvBlock3(initializer, weight_decay)

        self.conv1 = Conv2D(kernel_size=3,
                            filters=336,
                            strides=1,
                            use_bias=False,
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(weight_decay))

        self.conv2 = Conv2D(kernel_size=1,
                            filters=336,
                            strides=1,
                            use_bias=False,
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(weight_decay))

        self.conv3 = Conv2D(kernel_size=1,
                            filters=num_classes,
                            strides=1,
                            use_bias=False,
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(weight_decay))
        self.dropout1 = Dropout(0.5)
        self.dropout2 = Dropout(0.5)
        self.batchnorm1 = BatchNormalization(axis=-1)
        self.batchnorm2 = BatchNormalization(axis=-1)
        self.batchnorm3 = BatchNormalization(axis=-1)
        self.noise = GaussianNoise(0.3)
        self.avgpool = GlobalAveragePooling2D()

    def call(self, inputs, training=None, mask=None):
        output = self.convblock1(inputs, training=training)
        # print(output.shape)
        output = self.convblock2(output, training=training)
        # print(output.shape)
        output = self.convblock3(output, training=training)
        # print(output.shape)

        output = self.conv1(output)
        output = tf.nn.elu(self.batchnorm1(output))
        output = self.dropout1(output)
        # print(output.shape)

        output = self.conv2(output)
        output = tf.nn.elu(self.batchnorm2(output))
        output = self.dropout2(output)
        # print(output.shape)

        output = self.conv3(output)
        output = self.batchnorm3(output)
        output = self.avgpool(self.noise(output))
        # print(output.shape)
        return output


def main():
    model = VGGStyle(num_classes=10)
    input = tf.random_uniform((3, 128, 128, 2))
    model(input, training=True)
    model.summary()


if __name__ == '__main__':
    main()
