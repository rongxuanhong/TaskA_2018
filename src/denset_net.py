from tensorflow.keras.layers import BatchNormalization, \
    Conv2D, Activation, Dropout, AveragePooling2D, Concatenate, Input, Flatten, Dense, \
    GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from utils.utils import *


class DenseNet:
    def __init__(self, input_shape, n_classes, nb_layers, nb_dense_block, growth_rate,
                 weight_decay=1e-4,
                 theta=0.5, dropout_rate=0.5):
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.growth_rate = growth_rate
        self.nb_layers = nb_layers  # home many layers within denseblock
        self.nb_dense_block = nb_dense_block  # home many denseblock in model
        self.theta = theta  # a factor between 0-1 controls the number of feature maps from denseblock
        self.data_format = 'channels_last'
        # axis to be normalized (对channel axis操作)
        if self.data_format == 'channels_first':
            self.axis = 1
        else:
            self.axis = 3

    def Conv_2D(self, x, filters, kernel_size, name):
        return Conv2D(filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_uniform', use_bias=False,
                      kernel_regularizer=l2(self.weight_decay), name=name, data_format=self.data_format)(x)

    def conv_block(self, x, stage, branch, nb_filter,training):
        """
        Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        :param input:
        :param stage:index for dense block
        :param branch:layer index within each dense block
        :param nb_filter:k
        :return:
        """

        conv_name_base = 'conv' + str(stage) + '_' + str(branch)
        relu_name_base = 'relu' + str(stage) + '_' + str(branch)

        # 1x1 bottleneck 4k filters

        # x = BatchNormalization(axis=self.axis, gamma_regularizer=l2(self.weight_decay),
        #                        beta_regularizer=l2(self.weight_decay), name=conv_name_base + '_x1_bn')(x)
        x = BatchNormalization(axis=self.axis, epsilon=1.1e-5, name=conv_name_base + '_x1_bn', )(x)
        x = Activation('relu', name=relu_name_base + '_x1')(x)
        x = Conv2D(4 * nb_filter, 1, padding='same', use_bias=False, kernel_initializer='he_uniform',
                   kernel_regularizer=l2(self.weight_decay), name=conv_name_base + '_x1', data_format=self.data_format)(
            x)
        # if self.dropout_rate:
        # x = Dropout(self.dropout_rate)(x)

        # 3x3 con2d
        x = BatchNormalization(axis=self.axis, name=conv_name_base + '_x2_bn')(
            x)
        x = Activation('relu', name=relu_name_base + '_x2')(x)
        x = Conv2D(nb_filter, 3, padding='same', use_bias=False, kernel_initializer='he_uniform',
                   kernel_regularizer=l2(self.weight_decay), name=conv_name_base + '_x2', data_format=self.data_format)(
            x)
        x = Dropout(0.1)(x)

        return x

    # def conv_transpose_block(self, x, stage, nb_filter, kernel_size, strides):
    #     """
    #
    #     :param x:
    #     :param nb_filter:
    #     :param kernel_size:
    #     :param strides:
    #     :return:
    #     """
    #     deconv_name_base = 'deconv' + str(stage)
    #     x = Conv2DTranspose(nb_filter, kernel_size=kernel_size, strides=strides, use_bias=False,
    #                         kernel_initializer='he_uniform', padding='same',
    #                         kernel_regularizer=l2(self.weight_decay), name=deconv_name_base)(x)
    #     return x

    def transition_layers(self, x, stage, nb_filter,training):
        """
         a transition part contains bn relu 1x1conv and optional dropout ,followed by AveragePooling2D
        :param x:
        :param stage: index for denseblock
        :param nb_filter:  including feature maps from denseblock and itself
        :return:
        """
        conv_name_base = 'conv_' + str(stage) + '_tl'
        relu_name_base = 'relu_' + str(stage) + '_tl'
        pool_name_base = 'pool_' + str(stage)

        nb_filter = int(self.theta * nb_filter)  # 压缩feature map

        x = BatchNormalization(axis=self.axis, name=conv_name_base + '_bn')(x)
        x = Activation('relu', name=relu_name_base)(x)
        x = Conv2D(nb_filter, 1, padding='same', kernel_initializer='he_uniform', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), name=conv_name_base, data_format=self.data_format)(x)
        # if self.dropout_rate:
        x = Dropout(self.dropout_rate)(x,training=training)
        x = AveragePooling2D(pool_size=2, strides=2, name=pool_name_base, data_format=self.data_format)(
            x)  # non-overlap

        return x, nb_filter

    def dense_block(self, x, stage, nb_layers, nb_filter,training):
        """

        :param x:
        :param nb_layers: the number of layers of conv_block to append to the model.
        :param nb_filter: number of filters
        :return: x:keras model with nb_layers of conv_factory appended
        nb_filter:the number of feature maps on denseblock outputs
        """
        concat_feat = x
        for i in range(nb_layers):
            branch = i + 1
            x = self.conv_block(x, stage, branch, self.growth_rate,training)  ## simular to H function in paper
            concat_feat = Concatenate(axis=self.axis)(
                [concat_feat, x])  # concatenate feature maps from proceeding layers along feature axis or column
            nb_filter += self.growth_rate  #
        return concat_feat, nb_filter  # nb_filter=k0+k*nb_layers，denseblock has nb_filter output feature maps

    def build(self,training=True):
        input = Input(self.input_shape)
        # first convolution layer 3x3 conv
        x = self.Conv_2D(input, 2 * self.growth_rate, 3, name='conv_1')

        nb_filter = 2*self.growth_rate
        for i in range(self.nb_dense_block - 1):
            x, nb_filter = self.dense_block(x, i + 1, self.nb_layers, nb_filter,training)
            x, nb_filter = self.transition_layers(x, i + 1, nb_filter,training)

        x, nb_filter = self.dense_block(x, 5, self.nb_layers, nb_filter,training)

        x = GlobalAveragePooling2D()(x)

        # x = Flatten()(x)
        # x = Dense(54, kernel_initializer='he_uniform', name='fc1')(x)
        # x = BatchNormalization(name='fc1_bn', )(x)
        # x = Activation('relu')(x)
        # x = Dropout(self.dropout_rate)(x)
        # x = Dense(420, kernel_initializer='he_uniform', name='fc2')(x)
        # x = BatchNormalization(name='fc2_bn')(x)
        # x = Activation('relu')(x)
        # if self.dropout_rate:
        #     x = Dropout(self.dropout_rate)(x)
        logits = Dense(10, name='fc1')(x)

        model = Model(inputs=[input], outputs=[logits], name='densenet',)
        return model


if __name__ == '__main__':
    tf.enable_eager_execution()
    denset = DenseNet(input_shape=(100, 100, 3), n_classes=10, nb_layers=5,
                      nb_dense_block=5,
                      growth_rate=24)
    model = denset.build()
    print(tf.add_n(model.losses))
    # describe_model(model)
