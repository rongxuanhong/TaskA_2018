import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, AveragePooling2D, \
    Dense, Dropout, MaxPool2D, GlobalAveragePooling2D, Concatenate, Add, SeparableConv2D, SpatialDropout2D

from tensorflow.keras.regularizers import l2


class Conv2DBlock(tf.keras.Model):
    def __init__(self, filters, strides, block_index, weight_decay):
        super(Conv2DBlock, self).__init__()
        conv_name = "block" + str(block_index) + "_conv"
        bn_name = "block" + str(block_index) + "_bn"
        self.conv = Conv2D(filters,
                           kernel_size=(3, 3),
                           strides=strides,
                           kernel_regularizer=l2(weight_decay),
                           kernel_initializer='he_normal',
                           use_bias=False,
                           name=conv_name)

        self.batchnorm = BatchNormalization(name=bn_name)

    def call(self, inputs, training=None, mask=None):
        output = self.conv(inputs)
        output = self.batchnorm(output)
        return tf.nn.relu(output)


class SeparableConv2DBlock1(tf.keras.Model):
    def __init__(self, filters, block_index, weight_decay, relu_before_conv=False, pool=True,
                 has_residual=True, just_one_relu=False):
        super(SeparableConv2DBlock1, self).__init__()
        self.relu_before_conv = relu_before_conv
        self.has_residual = has_residual
        self.just_one_relu = just_one_relu
        self.pool = pool
        assert len(filters) == 2
        sepconv_name_base = "block" + str(block_index) + "_sepconv"
        conv_name_base = "block" + str(block_index) + "_conv"
        bn_name_base = "block" + str(block_index) + "_bn"
        pool_name_base = "block" + str(block_index) + "_pool"
        self.sepconv1 = SeparableConv2D(filters[0],
                                        kernel_size=(3, 3),
                                        use_bias=False,
                                        depthwise_regularizer=l2(weight_decay),
                                        pointwise_regularizer=l2(weight_decay),
                                        depthwise_initializer='he_normal',
                                        pointwise_initializer='he_normal',
                                        padding='same',
                                        name=sepconv_name_base + '1')
        self.sepconv2 = SeparableConv2D(filters[1],
                                        kernel_size=(3, 3),
                                        use_bias=False,
                                        padding='same',
                                        depthwise_regularizer=l2(weight_decay),
                                        pointwise_regularizer=l2(weight_decay),
                                        depthwise_initializer='he_normal',
                                        pointwise_initializer='he_normal',
                                        name=sepconv_name_base + '2')
        self.batchnorm1 = BatchNormalization(name=bn_name_base + '1')
        self.batchnorm2 = BatchNormalization(name=bn_name_base + '2')
        strides = 2

        if self.pool:
            self.maxpool = MaxPool2D((3, 3), strides=(2, 2), padding='same', name=pool_name_base)
        else:
            strides = 1

        if self.has_residual:
            self.batchnorm3 = BatchNormalization(name=bn_name_base + '3')
            self.conv = Conv2D(filters[1],
                               kernel_size=(1, 1),
                               strides=strides,
                               kernel_initializer='he_normal',
                               padding='same',
                               name=conv_name_base + "1x1")

    def call(self, inputs, training=None, mask=None):

        if self.relu_before_conv:
            output = self.sepconv1(tf.nn.relu(inputs))
            output = self.batchnorm1(output)

            output = self.sepconv2(tf.nn.relu(output))
            output = self.batchnorm2(output)

        else:  ## relu在后的有可能只有一次relu
            output = self.sepconv1(inputs)
            output = tf.nn.relu(self.batchnorm1(output))

            output = self.sepconv2(output)
            if self.just_one_relu:
                output = self.batchnorm2(output)
            else:
                output = tf.nn.relu(self.batchnorm2(output))

        if self.pool:
            output = self.maxpool(output)

        if self.has_residual:
            residual = self.batchnorm3(self.conv(inputs))
            output = Add()([residual, output])
        return output


class SeparableConv2DBlock2(tf.keras.Model):
    def __init__(self, filters, block_index, weight_decay):
        super(SeparableConv2DBlock2, self).__init__()
        sepconv_name_base = "block" + str(block_index) + "_sepconv"
        bn_name_base = "block" + str(block_index) + "_bn"
        self.sepconv1 = SeparableConv2D(filters,
                                        kernel_size=(3, 3),
                                        use_bias=False,
                                        padding='same',
                                        depthwise_regularizer=l2(weight_decay),
                                        pointwise_regularizer=l2(weight_decay),
                                        depthwise_initializer='he_normal',
                                        pointwise_initializer='he_normal',
                                        name=sepconv_name_base + '1')
        self.sepconv2 = SeparableConv2D(filters,
                                        kernel_size=(3, 3),
                                        use_bias=False,
                                        padding='same',
                                        depthwise_regularizer=l2(weight_decay),
                                        pointwise_regularizer=l2(weight_decay),
                                        depthwise_initializer='he_normal',
                                        pointwise_initializer='he_normal',
                                        name=sepconv_name_base + '2')
        self.sepconv3 = SeparableConv2D(filters,
                                        kernel_size=(3, 3),
                                        use_bias=False,
                                        depthwise_regularizer=l2(weight_decay),
                                        pointwise_regularizer=l2(weight_decay),
                                        depthwise_initializer='he_normal',
                                        pointwise_initializer='he_normal',
                                        padding='same',
                                        name=sepconv_name_base + '3')
        self.batchnorm1 = BatchNormalization(name=bn_name_base + '1')
        self.batchnorm2 = BatchNormalization(name=bn_name_base + '2')
        self.batchnorm3 = BatchNormalization(name=bn_name_base + '3')

    def call(self, inputs, training=None, mask=None):
        residual = inputs

        output = self.sepconv1(tf.nn.relu(inputs))
        output = self.batchnorm1(output)

        output = self.sepconv2(tf.nn.relu(output))
        output = self.batchnorm2(output)

        output = self.sepconv3(tf.nn.relu(output))
        output = self.batchnorm3(output)

        output = Add()([residual, output])
        return output


class Xception(tf.keras.Model):
    def __init__(self, num_classes, weight_decay=1e-4):
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.conv_block1 = Conv2DBlock(filters=32, strides=2, block_index=1, weight_decay=weight_decay)
        self.conv_block2 = Conv2DBlock(filters=64, strides=1, block_index=2, weight_decay=weight_decay)

        self.sep_conv_block3 = SeparableConv2DBlock1(filters=(128, 128), block_index=3, weight_decay=weight_decay,
                                                     just_one_relu=True)
        self.sep_conv_block4 = SeparableConv2DBlock1(filters=(256, 256), block_index=4, relu_before_conv=True,
                                                     weight_decay=weight_decay, pool=True)
        self.sep_conv_block5 = SeparableConv2DBlock1(filters=(728, 728), block_index=5, relu_before_conv=True,
                                                     weight_decay=weight_decay, pool=True)

        self.sep_conv_block2_6 = SeparableConv2DBlock2(filters=728, block_index=6, weight_decay=weight_decay)
        self.sep_conv_block2_7 = SeparableConv2DBlock2(filters=728, block_index=7, weight_decay=weight_decay)
        self.sep_conv_block2_8 = SeparableConv2DBlock2(filters=728, block_index=8, weight_decay=weight_decay)
        self.sep_conv_block2_9 = SeparableConv2DBlock2(filters=728, block_index=9, weight_decay=weight_decay)
        self.sep_conv_block2_10 = SeparableConv2DBlock2(filters=728, block_index=10, weight_decay=weight_decay)
        self.sep_conv_block2_11 = SeparableConv2DBlock2(filters=728, block_index=11, weight_decay=weight_decay)
        self.sep_conv_block2_12 = SeparableConv2DBlock2(filters=728, block_index=12, weight_decay=weight_decay)
        self.sep_conv_block2_13 = SeparableConv2DBlock2(filters=728, block_index=13, weight_decay=weight_decay)

        self.sep_conv_block14 = SeparableConv2DBlock1(filters=(728, 1024), block_index=14, relu_before_conv=True,
                                                      weight_decay=weight_decay)
        self.sep_conv_block15 = SeparableConv2DBlock1(filters=(1536, 2048), block_index=15, has_residual=False,
                                                      pool=False, weight_decay=weight_decay)

        self.avg_pool1 = GlobalAveragePooling2D(name='avg_pool1')
        self.avg_pool2 = GlobalAveragePooling2D(name='avg_pool2')
        self.avg_pool3 = GlobalAveragePooling2D(name='avg_pool3')
        self.fcn1 = Dense(512, kernel_initializer='he_normal', )
        self.dense = Dense(self.num_classes, name='prediction')
        self.dropout = Dropout(0.5)
        self.concate = Concatenate(axis=-1)

    def call(self, inputs, training=None, mask=None):
        output = self.conv_block1(inputs)
        output = self.conv_block2(output)

        output = self.sep_conv_block3(output)
        output = self.sep_conv_block4(output)
        output = self.sep_conv_block5(output)

        output = self.sep_conv_block2_6(output)
        output = self.sep_conv_block2_7(output)
        output = self.sep_conv_block2_8(output)
        output = self.sep_conv_block2_9(output)
        output = self.sep_conv_block2_10(output)
        output = self.sep_conv_block2_11(output)
        output = self.sep_conv_block2_12(output)
        output = self.sep_conv_block2_13(output)

        pool1 = self.avg_pool1(output)
        output = self.sep_conv_block14(output)
        pool2 = self.avg_pool2(output)
        output = self.sep_conv_block15(output)

        output = self.avg_pool3(output)
        output = self.concate([pool1, pool2, output])
        output = self.dropout(output, training=training)
        output = self.fcn1(output)
        logits = self.dense(output)

        return logits


if __name__ == '__main__':
    ## 由于输入尺寸较小 ，因此去掉了前面的几个池化层
    model = Xception(num_classes=10)
    input = tf.random_normal((3, 64, 157, 1))
    model(input)
    # model = tf.keras.applications.Xception(input_shape=(229, 229, 3))
    model.summary()
