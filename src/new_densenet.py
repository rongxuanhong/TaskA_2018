import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, GlobalAveragePooling2D, AveragePooling2D, Concatenate, \
    Conv2DTranspose, Add, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2


class Conv_block(tf.keras.Model):
    def __init__(self, stage, branch, nb_filter, data_format, bottleneck,
                 dropout_rate=0.2, weight_decay=1e-4):
        super(Conv_block, self).__init__()
        self.bottleneck = bottleneck
        inter_filter = nb_filter * 4
        conv_name_base = 'conv' + str(stage) + '_' + str(branch)
        axis = -1 if data_format == 'channels_last' else 1
        self.conv2 = Conv2D(nb_filter,
                            (3, 3),
                            padding='same',
                            use_bias=False,
                            data_format=data_format,
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(weight_decay),
                            name=conv_name_base + '_x2')
        # 初始化本模块所需要的op
        self.batchnorm1 = BatchNormalization(axis=axis, name=conv_name_base + '_x1_bn')
        self.dropout1 = Dropout(dropout_rate)
        if self.bottleneck:
            self.conv1 = Conv2D(inter_filter,
                                (1, 1),
                                padding='same',
                                use_bias=False,
                                data_format=data_format,
                                kernel_initializer='he_uniform',
                                kernel_regularizer=l2(weight_decay),
                                name=conv_name_base + '_x1')
            self.batchnorm2 = BatchNormalization(axis=axis, name=conv_name_base + '_x2_bn')

    def call(self, inputs, training=None, mask=None):
        output = self.batchnorm1(inputs, training=training)
        if self.bottleneck:
            output = self.conv1(tf.nn.relu(output))
            output = self.batchnorm2(output, training=training)
        output = self.conv2(tf.nn.relu(output))
        output = self.dropout1(output, training=training)
        return output


class TransitionBlock(tf.keras.Model):
    def __init__(self, stage, nb_filter, data_format, dropout_rate=0.0, weight_decay=1e-4):
        super(TransitionBlock, self).__init__()
        axis = -1 if data_format == 'channels_last' else 1
        conv_name_base = 'conv' + str(stage) + '_blk'
        pool_name_base = 'pool' + str(stage)
        ## 用于压缩的bottleneck
        self.conv1 = Conv2D(nb_filter, 1, padding='same', use_bias=False, kernel_initializer='he_uniform',
                            kernel_regularizer=l2(weight_decay), name=conv_name_base + '_x1')
        self.batchnorm = BatchNormalization(axis=axis)
        self.avg_pool = AveragePooling2D(data_format=data_format, name=pool_name_base)
        # self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=None, mask=None):
        output = self.batchnorm(inputs, training=training)
        output = self.conv1(tf.nn.relu(output))
        output = self.avg_pool(output)
        return output


class DenseBlock(tf.keras.Model):
    def __init__(self, stage, nb_layers, nb_filter, growth_rate, data_format, bottleneck, dropout_rate, weight_decay):
        super(DenseBlock, self).__init__()
        self.nb_layers = nb_layers
        self.bottleneck = bottleneck
        self.nb_filter = nb_filter
        self.growth_rate = growth_rate
        self.axis = -1 if data_format == 'channels_last' else 1
        self.blocks = []  # save each convblock in each denseblock
        for i in range(int(nb_layers)):
            branch = i + 1
            self.blocks.append(
                Conv_block(stage, branch, growth_rate, data_format, bottleneck, dropout_rate, weight_decay, ))

    def call(self, x, training=None, mask=None):
        concat_feat = x
        for i in range(self.nb_layers):
            x = self.blocks[i](x, training=training)  ## simular to H function in paper
            concat_feat = Concatenate(axis=self.axis)(
                [concat_feat, x])  # concatenate feature maps from proceeding layers along feature axis or column
            self.nb_filter += self.growth_rate  #
        return concat_feat, self.nb_filter  # nb_filter=k0+k*nb_layers，denseblock has nb_filter output feature map


class DenseNet(tf.keras.Model):
    def __init__(self, growth_rate, num_of_blocks,
                 output_classes, num_layers, data_format='channels_last',
                 bottleneck=True, compression=0.5, weight_decay=1e-4,
                 dropout_rate=0.2, ):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.num_of_blocks = num_of_blocks
        self.output_classes = output_classes
        self.num_layers = num_layers  # list tuple or integer
        self.data_format = data_format
        self.bottleneck = bottleneck
        self.compression = compression  # compression factor
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate

        self.conv1 = Conv2D(2 * self.growth_rate,
                            (3, 3),
                            padding='same',
                            use_bias=False,
                            data_format=self.data_format,
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(self.weight_decay),
                            name='conv1')

        self.conv2 = Conv2D(78,
                            1,
                            padding='same',
                            use_bias=False,
                            data_format=self.data_format,
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(self.weight_decay),
                            name='conv2')
        self.conv3 = Conv2D(74,
                            1,
                            padding='same',
                            use_bias=False,
                            data_format=self.data_format,
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(self.weight_decay),
                            name='conv3')
        self.conv4 = Conv2D(output_classes,
                            1,
                            padding='same',
                            use_bias=False,
                            data_format=self.data_format,
                            kernel_initializer='he_uniform',
                            kernel_regularizer=l2(self.weight_decay),
                            name='conv4')
        self.avg_pool = GlobalAveragePooling2D(data_format=self.data_format)
        self.dense = Dense(output_classes)

    def call(self, inputs, training=None, mask=None):
        output = self.conv1(inputs)
        nb_filter = 2 * self.growth_rate

        output_list = list()
        for i in range(self.num_of_blocks):
            output, nb_filter = DenseBlock(i + 1, self.num_layers, nb_filter, self.growth_rate, self.data_format,
                                           self.bottleneck, self.dropout_rate, self.weight_decay)(output,
                                                                                                  training=training)
            nb_filter = int(self.compression * nb_filter)
            output = TransitionBlock(i + 1, nb_filter, self.data_format, self.dropout_rate,
                                     self.weight_decay)(output, training=training)
            output_list.append((output, nb_filter))

        L1D = Conv2DTranspose(nb_filter, kernel_size=4, strides=4, use_bias=False,
                              kernel_initializer='he_uniform', padding='same',
                              kernel_regularizer=l2(self.weight_decay), name='convt1')(output)

        output = self.conv2(output_list[2][0])
        L2D = Conv2DTranspose(output_list[2][1], kernel_size=2, strides=2, use_bias=False,
                              kernel_initializer='he_uniform', padding='same',
                              kernel_regularizer=l2(self.weight_decay), name='convt2')(Add()([output, L1D]))
        output = self.conv3(output_list[1][0])
        L3D = Conv2DTranspose(output_list[1][1], kernel_size=4, strides=4, use_bias=False,
                              kernel_initializer='he_uniform', padding='same',
                              kernel_regularizer=l2(self.weight_decay), name='convt2')(Add()([output, L2D]))

        L0 = self.conv4(L3D)
        L = self.avg_pool(L0)  # gvp along frequency axis 64x6
        output = self.dense(L)
        return output


# class DenseNet(tf.keras.Model):
#     def __init__(self, n_classes, nb_layers, nb_dense_block, growth_rate, axis=3, dropout_rate=0.2,
#                  weight_decay=1e-4, theta=0.5):
#         super(DenseNet, self).__init__()
#         self.weight_decay = weight_decay
#         self.dropout_rate = dropout_rate
#         self.n_classes = n_classes
#         # self.input_shape = input_shape
#         self.growth_rate = growth_rate
#         self.axis = axis  # axis to be normalized (对channel axis操作)
#         self.nb_layers = nb_layers  # home many layers within denseblock
#         self.nb_dense_block = nb_dense_block  # home many denseblock in model
#         self.theta = theta  # a factor between 0-1 controls the number of feature maps from denseblock
#
#     def Conv_2D(self, x, filters, kernel_size, name):
#         return Conv2D(filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_uniform', use_bias=False,
#                       kernel_regularizer=l2(self.weight_decay), name=name)(x)
#
#     def conv_block(self, x, stage, branch, nb_filter, is_training):
#         """
#         Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
#         :param input:
#         :param stage:index for dense block
#         :param branch:layer index within each dense block
#         :param nb_filter:k
#         :return:
#         """
#
#         conv_name_base = 'conv' + str(stage) + '_' + str(branch)
#         relu_name_base = 'relu' + str(stage) + '_' + str(branch)
#
#         # 1x1 bottleneck 4k filters
#
#         x = BatchNormalization(axis=self.axis, name=conv_name_base + '_x1_bn')(x, training=is_training)
#         x = Activation('relu', name=relu_name_base + '_x1')(x)
#         x = Conv2D(4 * nb_filter, 1, padding='same', use_bias=False, kernel_initializer='he_uniform',
#                    kernel_regularizer=l2(self.weight_decay), name=conv_name_base + '_x1')(x)
#         if self.dropout_rate:
#             x = Dropout(self.dropout_rate)(x, training=is_training)
#
#         # 3x3 con2d
#         x = BatchNormalization(axis=self.axis, name=conv_name_base + '_x2_bn')(
#             x, training=is_training)
#         x = Activation('relu', name=relu_name_base + '_x2')(x)
#         x = Conv2D(nb_filter, 3, padding='same', use_bias=False, kernel_initializer='he_uniform',
#                    kernel_regularizer=l2(self.weight_decay), name=conv_name_base + '_x2')(x)
#         if self.dropout_rate:
#             x = Dropout(self.dropout_rate)(x, training=is_training)
#
#         return x
#
#     def conv_transpose_block(self, x, stage, nb_filter, kernel_size, strides):
#         """
#
#         :param x:
#         :param nb_filter:
#         :param kernel_size:
#         :param strides:
#         :return:
#         """
#         deconv_name_base = 'deconv' + str(stage)
#         x = Conv2DTranspose(nb_filter, kernel_size=kernel_size, strides=strides, use_bias=False,
#                             kernel_initializer='he_uniform', padding='same',
#                             kernel_regularizer=l2(self.weight_decay), name=deconv_name_base)(x)
#         return x
#
#     def transition_layers(self, x, stage, nb_filter, is_training):
#         """
#          a transition part contains bn relu 1x1conv and optional dropout ,followed by AveragePooling2D
#         :param x:
#         :param stage: index for denseblock
#         :param nb_filter:  including feature maps from denseblock and itself
#         :return:
#         """
#         conv_name_base = 'conv' + str(stage) + '_blk'
#         relu_name_base = 'relu' + str(stage) + '_blk'
#         pool_name_base = 'pool' + str(stage)
#
#         x = BatchNormalization(axis=self.axis, name=conv_name_base + '_bn')(x, training=is_training)
#         x = Activation('relu', name=relu_name_base)(x)
#         x = Conv2D(int(nb_filter * self.theta), 1, padding='same', kernel_initializer='he_uniform', use_bias=False,
#                    kernel_regularizer=l2(self.weight_decay), name=conv_name_base)(x)
#         # if self.dropout_rate:##去掉了dropout
#         #     x = Dropout(self.dropout_rate)(x)
#         x = AveragePooling2D(pool_size=2, strides=2, name=pool_name_base)(x)  # non-overlap
#         nb_filter = int(self.theta * nb_filter)
#
#         return x, nb_filter
#
#     def dense_block(self, x, stage, nb_layers, nb_filter, is_training):
#         """
#
#         :param x:
#         :param nb_layers: the number of layers of conv_block to append to the model.
#         :param nb_filter: number of filters
#         :return: x:keras model with nb_layers of conv_factory appended
#         nb_filter:the number of feature maps on denseblock outputs
#         """
#         concat_feat = x
#         for i in range(nb_layers):
#             branch = i + 1
#             x = self.conv_block(x, stage, branch, self.growth_rate, is_training)  ## simular to H function in paper
#             concat_feat = Concatenate(axis=self.axis)(
#                 [concat_feat, x])  # concatenate feature maps from proceeding layers along feature axis or column
#             nb_filter += self.growth_rate  #
#         return concat_feat, nb_filter  # nb_filter=k0+k*nb_layers，denseblock has nb_filter output feature maps
#
#     def call(self, inputs, training=None, mask=None):
#         # input = Input(self.input_shape)
#         # first convolution layer 3x3 conv
#         x = self.Conv_2D(inputs, 2 * self.growth_rate, 3, name='conv_1')
#         # print(x.shape)
#
#         # first DT
#         x, nb_filter = self.dense_block(x, 1, self.nb_layers, 2 * self.growth_rate, training)
#         L1, nb_filter1 = self.transition_layers(x, 1, nb_filter, training)
#
#         # print(x.shape)
#
#         # second DT
#         x, nb_filter = self.dense_block(L1, 2, self.nb_layers, nb_filter1, training)
#         L2, nb_filter2 = self.transition_layers(x, 2, nb_filter, training)
#         # print(x.shape)
#
#         # third DT
#         x, nb_filter = self.dense_block(L2, 3, self.nb_layers, nb_filter2, training)
#         L3, nb_filter3 = self.transition_layers(x, 3, nb_filter, training)
#         # print(x.shape)
#
#         # fourth DT
#         x, nb_filter = self.dense_block(L3, 4, self.nb_layers, nb_filter3, training)
#         L4, nb_filter4 = self.transition_layers(x, 4, nb_filter, training)
#         # print(x.shape)
#
#         # fifth DT
#         x, nb_filter = self.dense_block(L4, 5, self.nb_layers, nb_filter4, training)
#         L5, nb_filter5 = self.transition_layers(x, 5, nb_filter, training)
#
#         # print('L5的shape{}'.format(L5.shape))
#
#         L1D = self.conv_transpose_block(L5, 1, nb_filter5, kernel_size=4, strides=4)  # batch_sizex8x8x78
#
#         L3 = self.Conv_2D(L3, 78, 1, name='deconv_L1D')  # batch_sizex8x8x78
#
#         L2D = self.conv_transpose_block(Add()([L3, L1D]), 2, nb_filter3, kernel_size=2, strides=2)
#
#         L2 = self.Conv_2D(L2, 74, 1, name='deconv_L2D')  # batch_sizex16x16x72
#
#         L3D = self.conv_transpose_block(Add()([L2, L2D]), 3, nb_filter2, kernel_size=4, strides=4)
#         print(L3D.shape)
#
#         L0 = self.Conv_2D(L3D, self.n_classes, 1, name='conv_L3D')  # bottleneck layer 64x64x6# bottleneck layer 64x64x6
#         L = GlobalAveragePooling2D()(L0)  # gvp along frequency axis 64x6
#         output = Dense(self.n_classes)(L)
#         return output


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
    tf.enable_eager_execution()
    model = DenseNet(16, 5, 10, 5)
    rand_input = tf.random_uniform((3, 64, 64, 2))
    output = model(rand_input, training=False)
    loss = tf.add_n(model.losses)
    print(loss)
    # describe_model(model)
