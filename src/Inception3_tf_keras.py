from tensorflow.keras.layers import BatchNormalization, \
    Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, Dense, \
    GlobalAveragePooling2D, GaussianNoise
from utils.utils import *
from tensorflow.keras.regularizers import l2


class ConvBlockWithBN(tf.keras.Model):
    def __init__(self, filters, kernel_size, name, padding='same', strides=1, bn_axis=-1,
                 data_format='channels_last', dropout_rate=0.3):
        super(ConvBlockWithBN, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv = Conv2D(filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding=padding,
                           kernel_initializer='he_uniform',
                           data_format=data_format,
                           kernel_regularizer=l2(4e-5),
                           name=name)

        self.batchnorm = BatchNormalization(axis=bn_axis, )
        if dropout_rate:
            self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=None, mask=None):
        output = self.conv(inputs)
        output = tf.nn.relu(self.batchnorm(output, training))
        if self.dropout_rate:
            output = self.dropout(output, training)
        return output


class InceptionModule(tf.keras.Model):
    def __init__(self, filters, block, strides=1, data_format='channels_last'):
        super(InceptionModule, self).__init__()
        conv_base_name = 'inception_conv_' + block + '_branch'
        pool_base_name = 'inception_pool_' + block

        filter1, filter2, filter3, filter4 = filters
        bn_axis = -1 if data_format == 'channels_last' else 1

        self.conv_bn1 = ConvBlockWithBN(filter1, (1, 1), name=conv_base_name + '1x1', )

        self.conv_bn21 = ConvBlockWithBN(filter2[0], (1, 1), name=conv_base_name + '5x5_reduce', )
        self.conv_bn22 = ConvBlockWithBN(filter2[1], (5, 5), strides=strides, name=conv_base_name + '5x5', )

        self.conv_bn31 = ConvBlockWithBN(filter3[0], (1, 1), name=conv_base_name + 'double_3x3_reduce', )
        self.conv_bn32 = ConvBlockWithBN(filter3[1], (3, 3), name=conv_base_name + '3x3_1')
        self.conv_bn33 = ConvBlockWithBN(filter3[1], (3, 3), strides=strides, name=conv_base_name + '3x3_2', )

        self.avg_pool4 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same', name=pool_base_name)
        self.conv_bn4 = ConvBlockWithBN(filter4, (1, 1), name=conv_base_name + 'pool_proj', )

        self.concate = Concatenate(axis=bn_axis)

    def call(self, inputs, training=None, mask=None):
        x1 = self.conv_bn1(inputs, training=training)

        x2 = self.conv_bn21(inputs, training=training)
        x2 = self.conv_bn22(x2, training=training)

        x3 = self.conv_bn31(inputs, training=training)
        x3 = self.conv_bn32(x3, training=training)
        x3 = self.conv_bn33(x3, training=training)

        x4 = self.avg_pool4(inputs)
        x4 = self.conv_bn4(x4, training=training)

        output = self.concate([x1, x2, x3, x4])
        return output


class InceptionTransition1(tf.keras.Model):
    def __init__(self, filters, block, ):
        super(InceptionTransition1, self).__init__()
        assert len(filters) == 2
        conv_base_name = 'inception_conv_' + block + '_branch'
        pool_base_name = 'inception_pool_' + block

        filter1, filter2 = filters
        self.conv_bn1 = ConvBlockWithBN(filter1, (3, 3), strides=2, padding='valid', name=conv_base_name + '3x3', )

        self.conv_bn21 = ConvBlockWithBN(filter2[0], (1, 1), name=conv_base_name + 'double_3x3_reduce', )
        self.conv_bn22 = ConvBlockWithBN(filter2[1], (3, 3), name=conv_base_name + '3x3_1')
        self.conv_bn23 = ConvBlockWithBN(filter2[1], (3, 3), strides=2, padding='valid',
                                         name=conv_base_name + '3x3_2', )

        self.max_pool3 = MaxPooling2D(pool_size=(3, 3), strides=2, name=pool_base_name)

        self.concate = Concatenate(axis=-1)

    def call(self, inputs, training=None, mask=None):
        x1 = self.conv_bn1(inputs, training=training)

        x2 = self.conv_bn21(inputs, training=training)
        x2 = self.conv_bn22(x2, training=training)
        x2 = self.conv_bn23(x2, training=training)

        x3 = self.max_pool3(inputs)

        output = self.concate([x1, x2, x3])
        return output


class InceptionTransition2(tf.keras.Model):
    def __init__(self, filters, block, ):
        super(InceptionTransition2, self).__init__()
        assert len(filters) == 2
        conv_base_name = 'inception_conv_' + block + '_branch'
        pool_base_name = 'inception_pool_' + block

        filter1, filter2 = filters
        self.conv_bn11 = ConvBlockWithBN(filter1[0], (1, 1), name=conv_base_name + 'double_3x3_reduce', )
        self.conv_bn12 = ConvBlockWithBN(filter1[1], (3, 3), strides=2, padding='valid',
                                         name=conv_base_name + '3x3_1', )

        self.conv_bn21 = ConvBlockWithBN(filter2, (1, 1), name=conv_base_name + 'double_7x7_reduce', )
        self.conv_bn22 = ConvBlockWithBN(filter2, (1, 7), name=conv_base_name + '1x7')
        self.conv_bn23 = ConvBlockWithBN(filter2, (7, 1), name=conv_base_name + '7x1', )
        self.conv_bn24 = ConvBlockWithBN(filter2, (3, 3), strides=2, padding='valid', name=conv_base_name + '3x3_2', )

        self.max_pool3 = MaxPooling2D(pool_size=(3, 3), strides=2, name=pool_base_name)

        self.concate = Concatenate(axis=-1)

    def call(self, inputs, training=None, mask=None):
        x1 = self.conv_bn11(inputs, training=training)
        x1 = self.conv_bn12(x1, training=training)

        x2 = self.conv_bn21(inputs, training=training)
        x2 = self.conv_bn22(x2, training=training)
        x2 = self.conv_bn23(x2, training=training)
        x2 = self.conv_bn24(x2, training=training)

        x3 = self.max_pool3(inputs)

        output = self.concate([x1, x2, x3])
        return output


class InceptionWithFactorization(tf.keras.Model):
    def __init__(self, filters, block, data_format='channels_last'):
        super(InceptionWithFactorization, self).__init__()
        conv_base_name = 'inception_conv_' + block + '_branch'
        pool_base_name = 'inception_pool_' + block

        filter1, filter2, filter3, filter4 = filters
        bn_axis = -1 if data_format == 'channels_last' else 1

        self.conv_bn1 = ConvBlockWithBN(filter1, (1, 1), name=conv_base_name + '1x1', )

        self.conv_bn21 = ConvBlockWithBN(filter2[0], (1, 1), name=conv_base_name + '1x1_reduce', )
        self.conv_bn22 = ConvBlockWithBN(filter2[0], (1, 7), name=conv_base_name + '1x7', )
        self.conv_bn23 = ConvBlockWithBN(filter2[1], (7, 1), name=conv_base_name + '7x1', )

        self.conv_bn31 = ConvBlockWithBN(filter3[0], (1, 1), name=conv_base_name + 'double_1x1_reduce', )
        self.conv_bn32 = ConvBlockWithBN(filter3[0], (7, 1), name=conv_base_name + '7x1_1', )
        self.conv_bn33 = ConvBlockWithBN(filter3[0], (1, 7), name=conv_base_name + '1x7_1', )
        self.conv_bn34 = ConvBlockWithBN(filter3[0], (7, 1), name=conv_base_name + '7x1_2', )
        self.conv_bn35 = ConvBlockWithBN(filter3[1], (1, 7), name=conv_base_name + '1x7_2')

        self.avg_pool4 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same', name=pool_base_name)
        self.conv_bn4 = ConvBlockWithBN(filter4, (1, 1), name=conv_base_name + 'pool_proj', )

        self.concate = Concatenate(axis=bn_axis)

    def call(self, inputs, training=None, mask=None):
        x1 = self.conv_bn1(inputs, training=training)

        x2 = self.conv_bn21(inputs, training=training)
        x2 = self.conv_bn22(x2, training=training)
        x2 = self.conv_bn23(x2, training=training)

        x3 = self.conv_bn31(inputs, training=training)
        x3 = self.conv_bn32(x3, training=training)
        x3 = self.conv_bn33(x3, training=training)
        x3 = self.conv_bn34(x3, training=training)
        x3 = self.conv_bn35(x3, training=training)

        x4 = self.avg_pool4(inputs)
        x4 = self.conv_bn4(x4, training=training)

        output = self.concate([x1, x2, x3, x4])
        return output


class InceptionWithExpandFilters(tf.keras.Model):
    def __init__(self, filters, block, data_format='channels_last'):
        super(InceptionWithExpandFilters, self).__init__()
        assert len(filters) == 4
        conv_base_name = 'inception_conv_' + block + '_branch'
        pool_base_name = 'inception_pool_' + block

        filter1, filter2, filter3, filter4 = filters
        bn_axis = -1 if data_format == 'channels_last' else 1

        self.conv_bn1 = ConvBlockWithBN(filter1, (1, 1), name=conv_base_name + '1x1', )

        self.conv_bn21 = ConvBlockWithBN(filter2, (1, 1), name=conv_base_name + '1x3x1_reduce', )
        self.conv_bn22 = ConvBlockWithBN(filter2, (1, 3), name=conv_base_name + '1x3_2', )
        self.conv_bn23 = ConvBlockWithBN(filter2, (3, 1), name=conv_base_name + '3x1_2', )
        self.concate2 = Concatenate(axis=bn_axis)

        self.conv_bn31 = ConvBlockWithBN(filter3[0], (1, 1), name=conv_base_name + '3x3_reduce', )
        self.conv_bn32 = ConvBlockWithBN(filter3[1], (3, 3), name=conv_base_name + '3x3', )
        self.conv_bn33 = ConvBlockWithBN(filter3[1], (1, 3), name=conv_base_name + '1x3_3', )
        self.conv_bn34 = ConvBlockWithBN(filter3[1], (3, 1), name=conv_base_name + '3x1_3', )
        self.conv_bn35 = ConvBlockWithBN(filter3[1], (1, 7), name=conv_base_name + '1x7_2')
        self.concate3 = Concatenate(axis=bn_axis)

        self.avg_pool4 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same', name=pool_base_name)
        self.conv_bn4 = ConvBlockWithBN(filter4, (1, 1), name=conv_base_name + 'pool_proj', )

        self.concate = Concatenate(axis=bn_axis)

    def call(self, inputs, training=None, mask=None):
        x1 = self.conv_bn1(inputs, training=training)

        x2 = self.conv_bn21(inputs, training=training)
        x21 = self.conv_bn22(x2, training=training)
        x22 = self.conv_bn22(x2, training=training)
        x2 = self.concate2([x21, x22])

        x3 = self.conv_bn31(inputs, training=training)
        x3 = self.conv_bn32(x3, training=training)
        x31 = self.conv_bn33(x3, training=training)
        x32 = self.conv_bn34(x3, training=training)
        x3 = self.concate3([x31, x32])

        x4 = self.avg_pool4(inputs)
        x4 = self.conv_bn4(x4, training=training)

        output = self.concate([x1, x2, x3, x4])
        return output


class InceptionV3(tf.keras.Model):
    def __init__(self, num_classes, data_format='channels_last'):
        super(InceptionV3, self).__init__()

        self.conv_bn1 = ConvBlockWithBN(32, (3, 3), padding='valid', name='conv1', )
        # self.conv_bn2 = ConvBlockWithBN(32, (3, 3), padding='valid', name='conv2', dropout_rate=0)
        self.conv_bn3 = ConvBlockWithBN(64, (3, 3), name='conv3')

        # self.max_pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, data_format=data_format, name='maxpool1')

        # self.conv_bn4 = ConvBlockWithBN(80, (3, 3), padding='valid', name='conv4', dropout_rate=0.5)
        # self.conv_bn5 = ConvBlockWithBN(192, (3, 3), padding='valid', name='conv5', dropout_rate=0.5)

        self.max_pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, data_format=data_format, name='maxpool2')

        self.inception_module1 = InceptionModule([64, (48, 64), (64, 96), 32], 'mixed0')
        self.inception_module2 = InceptionModule([64, (48, 96), (64, 96), 64], 'mixed1')
        self.inception_module3 = InceptionModule([64, (48, 64), (64, 96), 64], 'mixed2')

        self.inception_transition1 = InceptionTransition1([384, (64, 96)], 'mixed3')

        self.inception_with_factorization1 = InceptionWithFactorization([192, (128, 192), (128, 192), 128], 'mixed4')
        self.inception_with_factorization2 = InceptionWithFactorization([192, (160, 192), (160, 192), 192], 'mixed5')
        self.inception_with_factorization3 = InceptionWithFactorization([192, (160, 192), (160, 192), 192], 'mixed6')
        self.inception_with_factorization4 = InceptionWithFactorization([192, (192, 192), (192, 192), 192], 'mixed7')

        self.inception_transition2 = InceptionTransition2([(192, 320), 192], 'mixed8')

        self.inception_with_expand_filters1 = InceptionWithExpandFilters([320, 384, (448, 384), 192], 'mixed9')
        self.inception_with_expand_filters2 = InceptionWithExpandFilters([320, 384, (448, 384), 192], 'mixed10')

        self.avg_pool1 = GlobalAveragePooling2D(data_format=data_format, name='global_avg_pool')
        self.avg_pool2 = GlobalAveragePooling2D(data_format=data_format, name='global_avg_pool')
        self.avg_pool3 = GlobalAveragePooling2D(data_format=data_format, name='global_avg_pool')
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(num_classes, name='predictions')
        self.concate = Concatenate(axis=-1)

    def call(self, inputs, training=None, mask=None):
        output = self.conv_bn1(inputs, training=training)
        # output = self.conv_bn2(output, training=training)
        output = self.conv_bn3(output, training=training)
        # output = self.conv_bn4(output, training=training)
        # output = self.conv_bn5(output, training=training)

        output = self.max_pool2(output)

        output = self.inception_module1(output, training=training)
        output = self.inception_module2(output, training=training)
        output = self.inception_module3(output, training=training)

        output = self.inception_transition1(output, training=training)
        output1 = self.avg_pool1(output)

        output = self.inception_with_factorization1(output, training=training)
        output = self.inception_with_factorization2(output, training=training)
        output = self.inception_with_factorization3(output, training=training)
        output = self.inception_with_factorization4(output, training=training)

        output = self.inception_transition2(output, training=training)
        output2 = self.avg_pool2(output)

        output = self.inception_with_expand_filters1(output, training=training)
        output = self.inception_with_expand_filters2(output, training=training)

        output = self.avg_pool3(output)
        output = self.concate([output1, output2, output])
        output = self.dense1(output)
        output = self.dense2(output)

        return output


if __name__ == '__main__':
    # tf.enable_eager_execution()
    model = InceptionV3(num_classes=10)
    rand_input = tf.random_uniform((3, 64, 64, 2))
    output = model(rand_input, training=True)
    model.summary()
