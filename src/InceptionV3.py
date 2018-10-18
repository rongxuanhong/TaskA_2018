from tensorflow.keras.layers import BatchNormalization, \
    Conv2D, AveragePooling2D, Input, Flatten, Dense, MaxPooling2D, Dropout, Concatenate, GlobalAveragePooling2D, \
    Activation
from tensorflow.keras.models import Model


def conv2DWithBN(x, filters, kernel_size, name, padding='same', strides=1, bn_axis=-1,
                 data_format='channels_last', ):
    x = Conv2D(filters,
               kernel_size=kernel_size,
               strides=strides,
               use_bias=False,  # 新增
               padding=padding,
               kernel_initializer='he_uniform',
               data_format=data_format,
               name=name)(x)
    x = BatchNormalization(bn_axis, )(x)
    return Activation('relu')(x)


def inception_module(input_tensor, filters, block, strides=1, data_format='channels_last'):
    assert len(filters) == 4
    conv_base_name = 'inception_conv_' + block + '_branch'
    pool_base_name = 'inception_pool_' + block

    filter1, filter2, filter3, filter4 = filters
    bn_axis = -1 if data_format == 'channels_last' else 1

    # first path
    x1 = conv2DWithBN(input_tensor, filter1, (1, 1), bn_axis=bn_axis, name=conv_base_name + '1x1', )

    # second path
    x2 = conv2DWithBN(input_tensor, filter2[0], (1, 1), bn_axis=bn_axis, name=conv_base_name + '5x5_reduce', )
    x2 = conv2DWithBN(x2, filter2[1], (5, 5), strides=strides, bn_axis=bn_axis, name=conv_base_name + '5x5', )

    # third path
    x3 = conv2DWithBN(input_tensor, filter3[0], (1, 1), bn_axis=bn_axis, name=conv_base_name + 'double_3x3_reduce', )
    x3 = conv2DWithBN(x3, filter3[1], (3, 3), bn_axis=bn_axis, name=conv_base_name + '3x3_1')
    x3 = conv2DWithBN(x3, filter3[1], (3, 3), strides=strides, bn_axis=bn_axis, name=conv_base_name + '3x3_2', )

    # forth path

    x4 = AveragePooling2D(pool_size=(3, 3),
                          strides=1,
                          padding='same',
                          name=pool_base_name)(input_tensor)

    x4 = conv2DWithBN(x4, filter4, (1, 1), bn_axis=bn_axis, name=conv_base_name + 'pool_proj', )

    x = Concatenate(axis=3)([x1, x2, x3, x4])
    print(x.shape)
    return x


def inception_module_transition1(input_tensor, filters, block, data_format='channels_last'):
    assert len(filters) == 2
    conv_base_name = 'inception_conv_' + block + '_branch'
    pool_base_name = 'inception_pool_' + block

    filter1, filter2 = filters
    bn_axis = -1 if data_format == 'channels_last' else 1

    # first path
    x1 = conv2DWithBN(input_tensor, filter1, (3, 3), strides=2, padding='valid', bn_axis=bn_axis,
                      name=conv_base_name + '3x3', )

    # second path
    x2 = conv2DWithBN(input_tensor, filter2[0], (1, 1), bn_axis=bn_axis, name=conv_base_name + 'double_3x3_reduce', )
    x2 = conv2DWithBN(x2, filter2[1], (3, 3), bn_axis=bn_axis, name=conv_base_name + '3x3_1')
    x2 = conv2DWithBN(x2, filter2[1], (3, 3), strides=2, padding='valid', bn_axis=bn_axis,
                      name=conv_base_name + '3x3_2', )

    # third path
    x3 = MaxPooling2D(pool_size=(3, 3), strides=2, name=pool_base_name)(input_tensor)

    x = Concatenate(axis=3)([x1, x2, x3])
    print(x.shape)
    return x


def inception_module_transition2(input_tensor, filters, block, data_format='channels_last'):
    assert len(filters) == 2
    conv_base_name = 'inception_conv_' + block + '_branch'
    pool_base_name = 'inception_pool_' + block

    filter1, filter2 = filters
    bn_axis = -1 if data_format == 'channels_last' else 1

    # first path
    x1 = conv2DWithBN(input_tensor, filter1[0], (1, 1), bn_axis=bn_axis, name=conv_base_name + 'double_3x3_reduce', )
    x1 = conv2DWithBN(x1, filter1[1], (3, 3), strides=2, padding='valid', bn_axis=bn_axis,
                      name=conv_base_name + '3x3_1', )

    # second path
    x2 = conv2DWithBN(input_tensor, filter2, (1, 1), bn_axis=bn_axis, name=conv_base_name + 'double_7x7_reduce', )
    x2 = conv2DWithBN(x2, filter2, (1, 7), bn_axis=bn_axis, name=conv_base_name + '1x7')
    x2 = conv2DWithBN(x2, filter2, (7, 1), bn_axis=bn_axis, name=conv_base_name + '7x1', )
    x2 = conv2DWithBN(x2, filter2, (3, 3), strides=2, padding='valid', bn_axis=bn_axis, name=conv_base_name + '3x3_2', )

    # third path
    x3 = MaxPooling2D(pool_size=(3, 3), strides=2, name=pool_base_name)(input_tensor)

    x = Concatenate(axis=3)([x1, x2, x3])
    print(x.shape)
    return x


def inception_module_with_factorization(input_tensor, filters, block, data_format='channels_last'):
    assert len(filters) == 4
    conv_base_name = 'inception_conv_' + block + '_branch'
    pool_base_name = 'inception_pool_' + block

    filter1, filter2, filter3, filter4 = filters
    bn_axis = -1 if data_format == 'channels_last' else 1

    # first path
    x1 = conv2DWithBN(input_tensor, filter1, (1, 1), bn_axis=bn_axis, name=conv_base_name + '1x1', )

    # second path
    x2 = conv2DWithBN(input_tensor, filter2[0], (1, 1), bn_axis=bn_axis, name=conv_base_name + '1x1_reduce', )
    x2 = conv2DWithBN(x2, filter2[0], (1, 7), bn_axis=bn_axis, name=conv_base_name + '1x7', )
    x2 = conv2DWithBN(x2, filter2[1], (7, 1), bn_axis=bn_axis, name=conv_base_name + '7x1', )

    # third path
    x3 = conv2DWithBN(input_tensor, filter3[0], (1, 1), bn_axis=bn_axis, name=conv_base_name + 'double_1x1_reduce', )
    x3 = conv2DWithBN(x3, filter3[0], (7, 1), bn_axis=bn_axis, name=conv_base_name + '7x1_1', )
    x3 = conv2DWithBN(x3, filter3[0], (1, 7), bn_axis=bn_axis, name=conv_base_name + '1x7_1', )
    x3 = conv2DWithBN(x3, filter3[0], (7, 1), bn_axis=bn_axis, name=conv_base_name + '7x1_2', )
    x3 = conv2DWithBN(x3, filter3[1], (1, 7), bn_axis=bn_axis, name=conv_base_name + '1x7_2')

    # forth path
    x4 = AveragePooling2D(pool_size=(3, 3),
                          strides=1,
                          padding='same',
                          name=pool_base_name)(input_tensor)

    x4 = conv2DWithBN(x4, filter4, (1, 1), bn_axis=bn_axis, name=conv_base_name + 'pool_proj', )

    x = Concatenate(axis=3)([x1, x2, x3, x4])
    print(x.shape)
    return x


def inception_module_with_expanded_filters(input_tensor, filters, block, data_format='channels_last'):
    assert len(filters) == 4
    conv_base_name = 'inception_conv_' + block + '_branch'
    pool_base_name = 'inception_pool_' + block

    filter1, filter2, filter3, filter4 = filters
    bn_axis = -1 if data_format == 'channels_last' else 1

    # first path
    x1 = conv2DWithBN(input_tensor, filter1, (1, 1), bn_axis=bn_axis, name=conv_base_name + '1x1', )

    # second path
    x2 = conv2DWithBN(input_tensor, filter2, (1, 1), bn_axis=bn_axis, name=conv_base_name + '1x3x1_reduce', )
    x_1 = conv2DWithBN(x2, filter2, (1, 3), bn_axis=bn_axis, name=conv_base_name + '1x3_2', )
    x_2 = conv2DWithBN(x2, filter2, (3, 1), bn_axis=bn_axis, name=conv_base_name + '3x1_2', )
    x2 = Concatenate(axis=bn_axis)([x_1, x_2])

    # third path
    x3 = conv2DWithBN(input_tensor, filter3[0], (1, 1), bn_axis=bn_axis, name=conv_base_name + '3x3_reduce', )
    x3 = conv2DWithBN(x3, filter3[1], (3, 3), bn_axis=bn_axis, name=conv_base_name + '3x3', )
    x_1 = conv2DWithBN(x3, filter3[1], (1, 3), bn_axis=bn_axis, name=conv_base_name + '1x3_3', )
    x_2 = conv2DWithBN(x3, filter3[1], (3, 1), bn_axis=bn_axis, name=conv_base_name + '3x1_3', )
    x3 = Concatenate(axis=bn_axis)([x_1, x_2])

    # forth path
    x4 = AveragePooling2D(pool_size=(3, 3),
                          strides=1,
                          padding='same',
                          name=pool_base_name)(input_tensor)

    x4 = conv2DWithBN(x4, filter4, (1, 1), bn_axis=bn_axis, name=conv_base_name + 'pool_proj', )

    x = Concatenate(axis=3)([x1, x2, x3, x4])
    return x


def InceptionV3(input_shape, num_classes, data_format='channels_last'):
    bn_axis = -1 if data_format == 'channels_last' else 1
    input = Input(input_shape)

    x = conv2DWithBN(input, 32, (3, 3), bn_axis=bn_axis, strides=2, padding='valid', name='conv1', )
    print(x.shape)
    x = conv2DWithBN(x, 32, (3, 3), bn_axis=bn_axis, padding='valid', name='conv2', )
    print(x.shape)
    x = conv2DWithBN(x, 64, (3, 3), bn_axis=bn_axis, name='conv3', )
    print(x.shape)
    x = MaxPooling2D(pool_size=(3, 3),
                     strides=2,
                     data_format=data_format,
                     name='maxpool1')(x)
    print(x.shape)
    x = conv2DWithBN(x, 80, (3, 3), bn_axis=bn_axis, padding='valid', name='conv4', )
    print(x.shape)
    x = conv2DWithBN(x, 192, (3, 3), bn_axis=bn_axis, padding='valid', strides=2, name='conv5', )
    print(x.shape)
    x = MaxPooling2D(pool_size=(3, 3),
                     strides=2,
                     data_format=data_format,
                     name='maxpool2')(x)

    print(x.shape)

    x = inception_module(x, [64, (48, 64), (64, 96), 32], 'mixed0')
    x = inception_module(x, [64, (48, 96), (64, 96), 64], 'mixed1')
    x = inception_module(x, [64, (48, 64), (64, 96), 64], 'mixed2')

    x = inception_module_transition1(x, [384, (64, 96)], 'mixed3')

    x = inception_module_with_factorization(x, [192, (128, 192), (128, 192), 128], 'mixed4')
    x = inception_module_with_factorization(x, [192, (160, 192), (160, 192), 192], 'mixed5')
    # x = inception_module_with_factorization(x, [192, (160, 192), (160, 192), 192], 'mixed6')
    x = inception_module_with_factorization(x, [192, (192, 192), (192, 192), 192], 'mixed7')

    x = inception_module_transition2(x, [(192, 320), 192], 'mixed8')

    x = inception_module_with_expanded_filters(x, [320, 384, (448, 384), 192], 'mixed9')
    x = inception_module_with_expanded_filters(x, [320, 384, (448, 384), 192], 'mixed10')

    x = GlobalAveragePooling2D(data_format=data_format, name='global_avg_pool')(x)
    print(x.shape)
    output = Dense(num_classes, activation='softmax',
                   kernel_initializer='he_uniform', name='predictions')(x)

    model = Model(input, [output], name='inception-V3')
    return model


def InceptionV3_small(input_shape, num_classes, data_format='channels_last'):
    bn_axis = -1 if data_format == 'channels_last' else 1
    input = Input(input_shape)

    x = conv2DWithBN(input, 32, (3, 3), bn_axis=bn_axis, padding='valid', name='conv1', )
    print(x.shape)
    # x = conv2DWithBN(x, 32, (3, 3), bn_axis=bn_axis, padding='valid', name='conv2', )
    # print(x.shape)
    x = conv2DWithBN(x, 64, (3, 3), bn_axis=bn_axis, name='conv3', )
    print(x.shape)
    # x = MaxPooling2D(pool_size=(3, 3),
    #                  strides=2,
    #                  data_format=data_format,
    #                  name='maxpool1')(x)
    # print(x.shape)
    # x = conv2DWithBN(x, 80, (3, 3), bn_axis=bn_axis, padding='valid', name='conv4', )
    # print(x.shape)
    # x = conv2DWithBN(x, 192, (3, 3), bn_axis=bn_axis, padding='valid', strides=2, name='conv5', )
    # print(x.shape)
    x = MaxPooling2D(pool_size=(3, 3),
                     strides=2,
                     data_format=data_format,
                     name='maxpool2')(x)

    print(x.shape)

    x = inception_module(x, [64, (48, 64), (64, 96), 32], 'mixed0')
    x = inception_module(x, [64, (48, 96), (64, 96), 64], 'mixed1')
    x = inception_module(x, [64, (48, 64), (64, 96), 64], 'mixed2')

    x = inception_module_transition1(x, [384, (64, 96)], 'mixed3')

    x = inception_module_with_factorization(x, [192, (128, 192), (128, 192), 128], 'mixed4')
    x = inception_module_with_factorization(x, [192, (160, 192), (160, 192), 192], 'mixed5')
    x = inception_module_with_factorization(x, [192, (160, 192), (160, 192), 192], 'mixed6')
    x = inception_module_with_factorization(x, [192, (192, 192), (192, 192), 192], 'mixed7')

    x = inception_module_transition2(x, [(192, 320), 192], 'mixed8')

    x = inception_module_with_expanded_filters(x, [320, 384, (448, 384), 192], 'mixed9')
    x = inception_module_with_expanded_filters(x, [320, 384, (448, 384), 192], 'mixed10')

    x = GlobalAveragePooling2D(data_format=data_format, name='global_avg_pool')(x)
    # x = Flatten()(x)
    output = Dense(num_classes, activation='softmax',
                   kernel_initializer='he_uniform', name='predictions')(x)

    model = Model(input, [output], name='inception-V3')
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
    model = InceptionV3_small(input_shape=(64, 64, 2), num_classes=10)
    describe_model(model)
