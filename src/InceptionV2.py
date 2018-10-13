from tensorflow.keras.layers import BatchNormalization, \
    Conv2D, AveragePooling2D, Input, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, GlobalAveragePooling2D, \
    Activation, SeparableConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def conv2DWithBN(x, filters, kernel_size, name, strides=1, bn_axis=-1, weight_decay=1e-4,
                 data_format='channels_last', ):
    x = Conv2D(filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               kernel_initializer='he_uniform',
               data_format=data_format,
               kernel_regularizer=l2(weight_decay),
               name=name)(x)
    x = BatchNormalization(bn_axis, )(x)
    return Activation('relu')(x)


def separableConv2DWithBN(x, filters, kernel_size, name, depth_multiplier, strides=1, bn_axis=-1, weight_decay=1e-4,
                          data_format='channels_last', ):
    x = SeparableConv2D(filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        kernel_initializer='he_uniform',
                        data_format=data_format,
                        kernel_regularizer=l2(weight_decay),
                        name=name)(x)
    x = BatchNormalization(bn_axis, )(x)
    return Activation('relu')(x)


def inception_module(input_tensor, filters, block, strides=1, pool_type='avg', pass_through=False, weight_decay=1e-4,
                     data_format='channels_last'):
    assert len(filters) == 4
    conv_base_name = 'inception_conv_' + block + '_branch'
    pool_base_name = 'inception_pool_' + block

    filter1, filter2, filter3, filter4 = filters
    bn_axis = -1 if data_format == 'channels_last' else 1

    # first path
    x1 = None
    if filter1:
        x1 = conv2DWithBN(input_tensor, filter1, (1, 1), bn_axis=bn_axis, name=conv_base_name + '1x1',
                          weight_decay=weight_decay)

    # second path
    x2 = conv2DWithBN(input_tensor, filter2[0], (1, 1), bn_axis=bn_axis, name=conv_base_name + '3x3_reduce',
                      weight_decay=weight_decay)
    x2 = conv2DWithBN(x2, filter2[1], (3, 3), strides=strides, bn_axis=bn_axis, name=conv_base_name + '3x3',
                      weight_decay=weight_decay)

    # third path
    x3 = conv2DWithBN(input_tensor, filter3[0], (1, 1), bn_axis=bn_axis, name=conv_base_name + 'double_3x3_reduce',
                      weight_decay=weight_decay)
    x3 = conv2DWithBN(x3, filter3[1], (3, 3), bn_axis=bn_axis, name=conv_base_name + '3x3_1', weight_decay=weight_decay)
    x3 = conv2DWithBN(x3, filter3[1], (3, 3), strides=strides, bn_axis=bn_axis, name=conv_base_name + '3x3_2',
                      weight_decay=weight_decay)

    # forth path
    if pool_type == 'max':
        x4 = MaxPooling2D(pool_size=(3, 3),
                          strides=strides,
                          padding='same',
                          name=pool_base_name)(input_tensor)

    else:
        x4 = AveragePooling2D(pool_size=(3, 3),
                              strides=1,
                              padding='same',
                              name=pool_base_name)(input_tensor)

    if not pass_through:
        x4 = conv2DWithBN(x4, filter4, (1, 1), bn_axis=bn_axis, name=conv_base_name + 'pool_proj',
                          weight_decay=weight_decay)

    if not filter1:
        x = Concatenate(axis=3)([x2, x3, x4])
    else:
        x = Concatenate(axis=3)([x1, x2, x3, x4])
    return x


def InceptionV2(input_shape, num_classes, weight_decay=1e-4, data_format='channels_last'):
    bn_axis = -1 if data_format == 'channels_last' else 1
    input = Input(input_shape)

    x = separableConv2DWithBN(input, 64, (7, 7), strides=2, depth_multiplier=8, name='conv1', weight_decay=weight_decay)

    x = MaxPooling2D(pool_size=(3, 3),
                     strides=2,
                     padding='same',
                     data_format=data_format,
                     name='maxpool1')(x)

    x = conv2DWithBN(x, 192, (3, 3), bn_axis=bn_axis, name='conv2', weight_decay=weight_decay)

    x = MaxPooling2D(pool_size=(3, 3),
                     strides=2,
                     padding='same',
                     data_format=data_format,
                     name='maxpool2')(x)

    x = inception_module(x, [64, (64, 64), (64, 96), 32], '3a')
    x = inception_module(x, [64, (64, 96), (64, 96), 64], '3b')
    x = inception_module(x, [0, (128, 160), (64, 96), 64], '3c', strides=2, pass_through=True, pool_type='max')

    x = inception_module(x, [224, (64, 96), (96, 128), 128], '4a')
    x = inception_module(x, [192, (96, 128), (96, 128), 128], '4b')
    x = inception_module(x, [160, (128, 160), (128, 160), 96], '4c')
    x = inception_module(x, [96, (128, 192), (160, 192), 96], '4d')
    x = inception_module(x, [0, (128, 192), (192, 256), 256], '4e', strides=2, pass_through=True, pool_type='max')

    x = inception_module(x, [352, (192, 320), (160, 224), 128], '5a')
    x = inception_module(x, [352, (192, 320), (192, 224), 128], '5b', pool_type='max')

    x = GlobalAveragePooling2D(data_format=data_format, name='global_avg')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    output = Dense(num_classes, activation='softmax',
                   kernel_initializer='he_uniform', name='fc1')(x)

    model = Model(input, [output], name='inception-V2')
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
    model = InceptionV2(input_shape=(224, 224, 3), num_classes=1000)

    describe_model(model)
