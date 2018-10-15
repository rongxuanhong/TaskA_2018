import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, AveragePooling2D, \
    Dense, Dropout, MaxPool2D, GlobalAveragePooling2D, Concatenate

from tensorflow.keras.regularizers import l2


class ConvBlock(tf.keras.Model):
    """bn->relu->-conv"""

    def __init__(self, num_filters, data_format, bottleneck, weight_decay=1e-4
                 , dropout_rate=0.):
        super(ConvBlock, self).__init__(name='conv_block')
        self.bottleneck = bottleneck
        axis = -1 if data_format == 'channels_last' else 1
        inter_filter = num_filters * 4
        self.conv2 = Conv2D(num_filters,
                            (3, 3),
                            padding='same',
                            use_bias=False,
                            data_format=data_format,
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(weight_decay))
        # 初始化本模块所需要的op
        self.batchnorm1 = BatchNormalization(axis=axis, epsilon=1.1e-5)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

        if self.bottleneck:
            self.conv1 = Conv2D(inter_filter,
                                (1, 1),
                                padding='same',
                                use_bias=False,
                                data_format=data_format,
                                kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay))
            self.batchnorm2 = BatchNormalization(axis=axis, epsilon=1.1e-5)

    def call(self, x, training=True, mask=None):

        output = self.batchnorm1(x, training=training)

        if self.bottleneck:
            output = self.conv1(tf.nn.relu(output))
            # output = self.dropout1(output, training=training) #暂时去除
            output = self.batchnorm2(output, training=training)

        output = self.conv2(tf.nn.relu(output))
        output = self.dropout2(output, training=training)
        return output


class TransitionBlock(tf.keras.Model):
    """transition block to reduce the number of filters"""

    def __init__(self, num_filters, data_format,
                 weight_decay=1e-4, dropout_rate=0.):
        super(TransitionBlock, self).__init__()
        axis = -1 if data_format == 'channels_last' else 1
        self.batchnorm = BatchNormalization(axis=axis, epsilon=1.1e-5)

        self.conv = Conv2D(num_filters,
                           (1, 1),
                           padding='same',
                           use_bias=False,
                           data_format=data_format,
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(weight_decay))
        self.avg_pool = AveragePooling2D(data_format=data_format)
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training=True, mask=None):
        #### 这里没有加 dropout ###
        output = self.batchnorm(x)
        output = self.conv(tf.nn.relu(output))
        # output = self.dropout(output, training=training) # 暂时去除
        output = self.avg_pool(output)
        return output


class DenseBlock(tf.keras.Model):
    def __init__(self, num_layers, growth_rate, data_format, bottleneck,
                 weight_decay=1e-4, dropout_rate=0.):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.axis = -1 if data_format == 'channels_last' else 1
        self.blocks = []  # save each convblock in each denseblock
        for _ in range(int(self.num_layers)):
            self.blocks.append(ConvBlock(growth_rate, data_format, bottleneck, weight_decay,
                                         dropout_rate))

    def call(self, x, training=True, mask=None):
        # concate each convblock within denseblock to get output of denseblock
        for i in range(int(self.num_layers)):
            output = self.blocks[i](x, training=training)
            x = tf.concat([x, output], axis=self.axis)

        return x


class DenseNet(tf.keras.Model):
    def __init__(self, depth_of_model, growth_rate, num_of_blocks,
                 output_classes, num_layers_in_each_block, data_format,
                 bottleneck=True, compression=0.5, weight_decay=1e-4,
                 dropout_rate=0.0, pool_initial=True, include_top=True):
        super(DenseNet, self).__init__()
        self.depth_of_model = depth_of_model  # valid when num_layers_in_each_block is integer
        self.growth_rate = growth_rate
        self.num_of_blocks = num_of_blocks
        self.output_classes = output_classes
        self.num_layers_in_each_block = num_layers_in_each_block  # list tuple or integer
        self.data_format = data_format
        self.bottleneck = bottleneck
        self.compression = compression  # compression factor
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.pool_initial = pool_initial
        self.include_top = include_top

        # 决定每个block的层数
        if isinstance(self.num_layers_in_each_block, list) or isinstance(
                self.num_layers_in_each_block, tuple):  # 指定每个blocks的层数
            self.num_layers_in_each_block = list(self.num_layers_in_each_block)
        else:
            if self.num_layers_in_each_block == -1:  # 由模型深度决定每个block的层数
                if self.num_of_blocks != 3:
                    raise ValueError(
                        'Number of blocks must be 3 if num_layers_in_each_block is -1')
                if (self.depth_of_model - 4) % 3 == 0:
                    num_layers = (self.depth_of_model - 4) / 3
                    if self.bottleneck:
                        num_layers //= 2
                    self.num_layers_in_each_block = [num_layers] * self.num_of_blocks
                else:
                    raise ValueError("Depth must be 3N+4 if num_layer_in_each_block=-1")
            else:  # 每个blocks的层数相同
                self.num_layers_in_each_block = [self.num_layers_in_each_block] * self.num_of_blocks

        axis = -1 if data_format == 'channels_last' else 1

        # setting the filters and stride of the initial covn layer.
        if self.pool_initial:
            init_filters = (3, 3)
            stride = (2, 2)
        else:
            init_filters = (3, 3)
            stride = (1, 1)
        self.num_filters = 2 * self.growth_rate

        # 定义第一个conv以及pool
        self.conv1 = Conv2D(self.num_filters,
                            init_filters,
                            strides=stride,
                            padding='same',
                            use_bias=False,
                            data_format=self.data_format,
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(self.weight_decay))

        if self.pool_initial:
            self.pool1 = MaxPool2D(pool_size=(3, 3),
                                   strides=(2, 2),
                                   padding='same',
                                   data_format=self.data_format)
            self.batchnorm1 = BatchNormalization(axis=axis, epsilon=1.1e-5)
        self.batchnorm2 = BatchNormalization(axis=axis, epsilon=1.1e-5)

        # last pool and fc layer
        if self.include_top:  # is need top layer
            self.last_pool = GlobalAveragePooling2D(data_format=self.data_format)
            self.classifier = Dense(self.output_classes)

        # calculate the number of filters after each denseblock
        num_filters_after_each_block = [self.num_filters]
        for i in range(1, self.num_of_blocks):
            temp_num_filters = num_filters_after_each_block[i - 1] + \
                               self.growth_rate * self.num_layers_in_each_block[i - 1]
            num_filters_after_each_block.append(int(temp_num_filters * self.compression))  # compress filters

        # dense block initialization
        self.dense_block = []
        self.transition_blocks = []
        for i in range(self.num_of_blocks):
            self.dense_block.append(DenseBlock(self.num_layers_in_each_block[i],
                                               self.growth_rate,
                                               self.data_format,
                                               self.weight_decay,
                                               self.dropout_rate))
            if i + 1 < self.num_of_blocks:
                self.transition_blocks.append(TransitionBlock(num_filters_after_each_block[i],
                                                              self.data_format,
                                                              self.weight_decay,
                                                              self.dropout_rate))

    # def call(self, x, training=True, mask=None):
    #     """ general modelling of DenseNet"""
    #     output = self.conv1(x)
    #
    #     if self.pool_initial:
    #         output = self.batchnorm1(output, training=training)
    #         output = self.pool1(tf.nn.relu(output))
    #
    #     for i in range(self.num_of_blocks - 1):
    #         output = self.dense_block[i](output, training=training)
    #         output = self.transition_blocks[i](output, training=training)
    #
    #     output = self.dense_block[self.num_of_blocks - 1](output, training=training)  # output of the last denseblock
    #     output = self.batchnorm2(output, training=training)
    #     output = tf.nn.relu(output)
    #
    #     if self.include_top:
    #         output = self.last_pool(output)
    #         output = self.classifier(output)
    #
    #     return output
    def call(self, x, training=True, mask=None):
        """ general modelling of DenseNet"""
        output = self.conv1(x)

        if self.pool_initial:
            output = self.batchnorm1(output, training=training)
            output = self.pool1(tf.nn.relu(output))

        avg_pool_list = list()
        for i in range(self.num_of_blocks - 1):
            output = self.dense_block[i](output, training=training)
            output = self.transition_blocks[i](output, training=training)
            if i != 0:
                avg_pool_list.append(GlobalAveragePooling2D()(output))

        output = self.dense_block[self.num_of_blocks - 1](output, training=training)  # output of the last denseblock
        output = self.batchnorm2(output, training=training)
        output = tf.nn.relu(output)
        avg_pool_list.append(GlobalAveragePooling2D()(output))
        output = Concatenate()(avg_pool_list)

        output = Dense(self.output_classes, )(output)

        # if self.include_top:
        #     output = self.last_pool(output)
        #     output = self.classifier(output)

        return output


def main():
    tf.enable_eager_execution()
    model = DenseNet(7, 16, 5, 10, 5, 'channels_last', True)
    rand_input = tf.random_uniform((3, 100, 100, 3))
    output = model(rand_input)
    print(tf.add_n(model.losses))
    # from utils.utils import describe_model
    # model = tf.keras.models.Model(inputs=[input], outputs=[output], name='densenet',)
    # describe_model(model)
    # print_num_of_total_parameters(model.trainable_variables)


if __name__ == '__main__':
    main()
