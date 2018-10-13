from tensorflow.keras.layers import BatchNormalization, \
    Conv2D, AveragePooling2D, Input, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def inception_module(input_tensor, filters, block, weight_decay=1e-4, data_format='channels_last'):
    assert len(filters) == 4
    conv_base_name = 'inception_conv_' + block + '_branch'
    pool_base_name = 'inception_pool_' + block

    filter1, filter2, filter3, filter4 = filters
    assert len(filter2) == 2
    assert len(filter3) == 2

    axis = -1 if data_format == 'channels_last' else 1
    # first path
    x1 = Conv2D(filter1,
                kernel_size=(1, 1),
                padding='same',
                data_format=data_format,
                kernel_regularizer=l2(weight_decay),
                name=conv_base_name + '1x1')(input_tensor)
    # second path
    x2 = Conv2D(filter2[0],
                kernel_size=(1, 1),
                padding='same',
                data_format=data_format,
                kernel_regularizer=l2(weight_decay),
                name=conv_base_name + '3x3_reduce')(input_tensor)

    x2 = Conv2D(filter2[1],
                kernel_size=(3, 3),
                padding='same',
                data_format=data_format,
                kernel_regularizer=l2(weight_decay),
                name=conv_base_name + '3x3')(x2)

    # third path
    x3 = Conv2D(filter3[0],
                kernel_size=(1, 1),
                padding='same',
                data_format=data_format,
                kernel_regularizer=l2(weight_decay),
                name=conv_base_name + '5x5_reduce')(input_tensor)

    x3 = Conv2D(filter3[1],
                kernel_size=(5, 5),
                padding='same',
                data_format=data_format,
                kernel_regularizer=l2(weight_decay),
                name=conv_base_name + '5x5')(x3)

    # forth path
    x4 = MaxPooling2D(pool_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      name=pool_base_name)(input_tensor)

    x4 = Conv2D(filter4,
                kernel_size=(1, 1),
                padding='same',
                data_format=data_format,
                kernel_regularizer=l2(weight_decay),
                name=conv_base_name + 'pool_proj')(x4)

    x = Concatenate(axis=axis)([x1, x2, x3, x4])
    return x


def intermediate_classifier(input_tensor, num_classes, index, weight_decay=1e-4, data_format='channels_last'):
    conv_base_name = 'softmax_conv' + str(index)
    pool_base_name = 'softmax_avgpool' + str(index)
    dense_base_name = 'softmax_dense' + str(index)

    x = AveragePooling2D(pool_size=(5, 5),
                         strides=3,
                         padding='same',
                         data_format=data_format,
                         name=pool_base_name)(input_tensor)

    x = Conv2D(128,
               kernel_size=(1, 1),
               padding='same',
               data_format=data_format,
               kernel_regularizer=l2(weight_decay),
               name=conv_base_name + '1x1', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name=dense_base_name + 'fc1')(x)
    x = Dropout(0.7)(x)
    x = Dense(num_classes, activation='relu', name=dense_base_name + 'fc2')(x)
    return x


def GoogLeNet(input_shape, num_classes, weight_decay=1e-4, data_format='channels_last'):
    input = Input(input_shape)
    x = Conv2D(filters=64,
               kernel_size=(7, 7),
               strides=(2, 2),
               padding='same',
               data_format=data_format,
               kernel_regularizer=l2(weight_decay),
               name='conv1')(input)

    x = MaxPooling2D(pool_size=(3, 3),
                     strides=2,
                     padding='same',
                     data_format=data_format,
                     name='maxpool1')(x)

    x = Conv2D(filters=192,
               kernel_size=(3, 3),
               padding='same',
               data_format=data_format,
               kernel_regularizer=l2(weight_decay),
               name='conv2')(x)

    x = MaxPooling2D(pool_size=(3, 3),
                     strides=2,
                     padding='same',
                     data_format=data_format,
                     name='maxpool2')(x)

    x = inception_module(x, [64, (96, 128), (16, 32), 32], '3a')
    x = inception_module(x, [128, (128, 192), (32, 96), 64], '3b')

    x = MaxPooling2D(pool_size=(3, 3),
                     strides=2,
                     padding='same',
                     data_format=data_format,
                     name='maxpool3')(x)

    x1 = inception_module(x, [192, (96, 208), (16, 48), 64], '4a')
    x = inception_module(x1, [160, (112, 224), (24, 64), 64], '4b')
    x = inception_module(x, [128, (128, 256), (24, 64), 64], '4c')
    x2 = inception_module(x, [112, (144, 288), (32, 64), 64], '4d')
    x = inception_module(x2, [256, (160, 320), (32, 128), 128], '4e')

    # 后面被证明 它们只是只能充当正则化的效果，不需要加，加入bn dropout更好
    # output1 = intermediate_classifier(x1, num_classes, 0)
    # output2 = intermediate_classifier(x2, num_classes, 1)

    x = MaxPooling2D(pool_size=(3, 3),
                     strides=2,
                     padding='same',
                     data_format=data_format,
                     name='maxpool4')(x)

    x = inception_module(x, [256, (160, 320), (32, 128), 128], '5a')
    x = inception_module(x, [384, (192, 384), (48, 128), 128], '5b')

    x = AveragePooling2D(pool_size=(7, 7),
                         strides=1,
                         padding='valid',
                         data_format=data_format,
                         name='avg_pool1')(x)
    x = Dropout(0.4)(x)
    # linear layer followed by softmax
    x = Flatten()(x)
    output= Dense(num_classes, activation='softmax', name='fc1')(x)

    model = Model(input, [output], name='GoogLenet')
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
    model = GoogLeNet(input_shape=(224, 224, 3), num_classes=1000)

    describe_model(model)
