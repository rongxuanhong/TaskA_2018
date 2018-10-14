from tensorflow.keras.layers import Conv2D, Activation, Dropout, GlobalAveragePooling2D, AveragePooling2D, Concatenate, \
    Input, Conv2DTranspose, Add, BatchNormalization, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


class DenseNet:
    def __init__(self, input_shape, n_classes, nb_layers, nb_dense_block, growth_rate, axis=-1,
                 dropout_rate=0.5,
                 weight_decay=1e-5,
                 theta=0.5):
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.growth_rate = growth_rate
        self.axis = axis  # axis to be normalized (对channel axis操作)
        self.nb_layers = nb_layers  # home many layers within denseblock
        self.nb_dense_block = nb_dense_block  # home many denseblock in model
        self.theta = theta  # a factor between 0-1 controls the number of feature maps from denseblock

    def Conv_2D(self, x, filters, kernel_size, name):
        return Conv2D(filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', use_bias=False,
                      kernel_regularizer=l2(self.weight_decay), name=name)(x)

    def conv_block(self, x, stage, branch, nb_filter):
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

        x = BatchNormalization(axis=self.axis, epsilon=1.1e-5, name=conv_name_base + '_x1_bn')(
            x)
        x = Activation('relu', name=relu_name_base + '_x1')(x)
        x = Conv2D(4 * nb_filter, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(self.weight_decay), name=conv_name_base + '_x1')(x)

        # 3x3 con2d
        x = BatchNormalization(axis=self.axis, epsilon=1.1e-5, name=conv_name_base + '_x2_bn')(
            x)
        x = Activation('relu', name=relu_name_base + '_x2')(x)
        x = Conv2D(nb_filter, 3, padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(self.weight_decay), name=conv_name_base + '_x2')(x)
        if self.dropout_rate:
            x = Dropout(self.dropout_rate)(x)

        return x

    def conv_transpose_block(self, x, stage, nb_filter, kernel_size, strides):
        """

        :param x:
        :param nb_filter:
        :param kernel_size:
        :param strides:
        :return:
        """
        deconv_name_base = 'deconv' + str(stage)
        x = Conv2DTranspose(nb_filter, kernel_size=kernel_size, strides=strides, use_bias=False,
                            kernel_initializer='he_normal', padding='same',
                            kernel_regularizer=l2(self.weight_decay), name=deconv_name_base)(x)
        return x

    def transition_layers(self, x, stage, nb_filter):
        """
         a transition part contains bn relu 1x1conv and optional dropout ,followed by AveragePooling2D
        :param x:
        :param stage: index for denseblock
        :param nb_filter:  including feature maps from denseblock and itself
        :return:
        """
        conv_name_base = 'conv' + str(stage) + '_blk'
        relu_name_base = 'relu' + str(stage) + '_blk'
        pool_name_base = 'pool' + str(stage)

        x = BatchNormalization(axis=self.axis, epsilon=1.1e-5, name=conv_name_base + '_bn')(
            x)
        x = Activation('relu', name=relu_name_base)(x)
        x = Conv2D(int(nb_filter * self.theta), 1, padding='same', kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=l2(self.weight_decay), name=conv_name_base)(x)
        if self.dropout_rate:
            x = Dropout(self.dropout_rate)(x)
        x = AveragePooling2D(pool_size=2, strides=2, name=pool_name_base)(x)  # non-overlap
        nb_filter = int(self.theta * nb_filter)

        return x, nb_filter

    def dense_block(self, x, stage, nb_layers, nb_filter):
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
            x = self.conv_block(x, stage, branch, self.growth_rate)  ## simular to H function in paper
            concat_feat = Concatenate(axis=self.axis)(
                [concat_feat, x])  # concatenate feature maps from proceeding layers along feature axis or column
            nb_filter += self.growth_rate  #
        return concat_feat, nb_filter  # nb_filter=k0+k*nb_layers，denseblock has nb_filter output feature maps

    def build(self):

        input = Input(self.input_shape)
        # first convolution layer 3x3 conv
        x = self.Conv_2D(input, 2 * self.growth_rate, 3, name='conv_1')
        # print(x.shape)

        # first DT
        x, nb_filter = self.dense_block(x, 1, self.nb_layers, self.growth_rate)
        L1, nb_filter1 = self.transition_layers(x, 1, nb_filter)

        # print("DT1的shape:", L1.shape)

        # second DT
        x, nb_filter = self.dense_block(L1, 2, self.nb_layers, nb_filter1)
        L2, nb_filter2 = self.transition_layers(x, 2, nb_filter)

        # print("DT2的shape:", L2.shape)

        # third DT
        x, nb_filter = self.dense_block(L2, 3, self.nb_layers, nb_filter2)
        L3, nb_filter3 = self.transition_layers(x, 3, nb_filter)
        # print("DT3的shape:", L3.shape)

        # fourth DT
        x, nb_filter = self.dense_block(L3, 4, self.nb_layers, nb_filter3)
        L4, nb_filter4 = self.transition_layers(x, 4, nb_filter)

        # print("DT4的shape:", L4.shape)
        # fifth DT
        x, nb_filter = self.dense_block(L4, 5, self.nb_layers, nb_filter4)
        L5, nb_filter5 = self.transition_layers(x, 5, nb_filter)

        # print("DT5的shape:", L5.shape)

        # L1D = self.conv_transpose_block(L5, 1, nb_filter5, kernel_size=4, strides=4)  # batch_sizex8x8x78
        #
        # L3 = self.Conv_2D(L3, 78, 1, name='deconv_L1D')  # batch_sizex8x8x72
        #
        # L2D = self.conv_transpose_block(Add()([L3, L1D]), 2, nb_filter3, kernel_size=2, strides=2)
        #
        # L2 = self.Conv_2D(L2, 72, 1, name='deconv_L2D')
        #
        # L3D = self.conv_transpose_block(Add()([L2, L2D]), 3, nb_filter2, kernel_size=4, strides=4)
        #
        # L0 = self.Conv_2D(L3D, self.n_classes, 1, name='conv_L3D')  # bottleneck layer 64x64x6# bottleneck layer 64x64x6
        x1 = GlobalAveragePooling2D()(L2)
        x2 = GlobalAveragePooling2D()(L3)
        x3 = GlobalAveragePooling2D()(L4)
        x4 = GlobalAveragePooling2D()(L5)

        x = Concatenate(axis=-1)([x1, x2, x3, x4])
        # print(x.shape)

        output = Dense(10, name='prediction')(x)
        print(output.shape)

        model = Model(input, output, name='densenet')
        return model


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


if __name__ == '__main__':
    # import tensorflow as tf
    # tf.enable_eager_execution()
    model = DenseNet((100, 100, 3), 10, 5, 5, 24, dropout_rate=0.5)
    model = model.build()
    # l2_loss = tf.add_n(model.losses)
    # print(l2_loss)
    describe_model(model)