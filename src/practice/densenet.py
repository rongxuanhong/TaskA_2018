from tensorflow.keras.layers import Conv2D, Activation, Dropout, GlobalAveragePooling2D, AveragePooling2D, Concatenate, Input, \
    Conv2DTranspose, Add,BatchNormalization
from keras.regularizers import l2
from keras.models import Model
import tensorflow as tf


class DenseNet:
    def __init__(self, input_shape, n_classes, nb_layers, nb_dense_block, threshold, growth_rate, axis=3,
                 weight_decay=1e-4,
                 theta=0.5):
        self.weight_decay = weight_decay
        self.dropout_rate = None
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.growth_rate = growth_rate
        self.axis = axis  # axis to be normalized (对channel axis操作)
        self.nb_layers = nb_layers  # home many layers within denseblock
        self.nb_dense_block = nb_dense_block  # home many denseblock in model
        self.theta = theta  # a factor between 0-1 controls the number of feature maps from denseblock
        self.threshold = threshold

    def Conv_2D(self, x, filters, kernel_size, name):
        return Conv2D(filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_uniform', use_bias=False,
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

        x = BatchNormalization(axis=self.axis, gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay), name=conv_name_base + '_x1_bn')(
            x)
        x = Activation('relu', name=relu_name_base + '_x1')(x)
        x = Conv2D(4 * nb_filter, 1, padding='same', use_bias=False, kernel_initializer='he_uniform',
                   kernel_regularizer=l2(self.weight_decay), name=conv_name_base + '_x1')(x)
        if self.dropout_rate:
            x = Dropout(self.dropout_rate)(x)

        # 3x3 con2d
        x = BatchNormalization(axis=self.axis, gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay), name=conv_name_base + '_x2_bn')(
            x)
        x = Activation('relu', name=relu_name_base + '_x2')(x)
        x = Conv2D(nb_filter, 3, padding='same', use_bias=False, kernel_initializer='he_uniform',
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
                            kernel_initializer='he_uniform', padding='same',
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

        x = BatchNormalization(axis=self.axis, gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay), name=conv_name_base + '_bn')(
            x)
        x = Activation('relu', name=relu_name_base)(x)
        x = Conv2D(int(nb_filter * self.theta), 1, padding='same', kernel_initializer='he_uniform', use_bias=False,
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
        print("第一层卷基层shape: %s", x.shape)

        # first DT
        x, nb_filter = self.dense_block(x, 1, self.nb_layers, self.growth_rate)
        L1, nb_filter1 = self.transition_layers(x, 1, nb_filter)

        print("L1 的shape:%s",L1.shape)

        # second DT
        x, nb_filter = self.dense_block(L1, 2, self.nb_layers, nb_filter1)
        L2, nb_filter2 = self.transition_layers(x, 2, nb_filter)

        # third DT
        x, nb_filter = self.dense_block(L2, 3, self.nb_layers, nb_filter2)
        L3, nb_filter3 = self.transition_layers(x, 3, nb_filter)

        # fourth DT
        x, nb_filter = self.dense_block(L3, 4, self.nb_layers, nb_filter3)
        L4, nb_filter4 = self.transition_layers(x, 4, nb_filter)

        # fifth DT
        x, nb_filter = self.dense_block(L4, 5, self.nb_layers, nb_filter4)
        L5, nb_filter5 = self.transition_layers(x, 5, nb_filter)

        print('L5的shape{}'.format(L5.shape))


        L1D = self.conv_transpose_block(L5, 1, nb_filter5, kernel_size=4, strides=4)  # batch_sizex8x8x78

        L3 = self.Conv_2D(L3, 78, 1, name='deconv_L1D')  # batch_sizex8x8x72

        L2D = self.conv_transpose_block(Add()([L3, L1D]), 2, nb_filter3, kernel_size=2, strides=2)

        L2 = self.Conv_2D(L2, 72, 1, name='deconv_L2D')

        L3D = self.conv_transpose_block(Add()([L2, L2D]), 3, nb_filter2, kernel_size=4, strides=4)

        L0 = self.Conv_2D(L3D, self.n_classes, 1, name='conv_L3D')  # bottleneck layer 64x64x6# bottleneck layer 64x64x6
        # L = GlobalAveragePooling2D()(L0)  # gvp along frequency axis 64x6


        model = Model(inputs=[input], outputs=[L0], name='densenet')
        return model
