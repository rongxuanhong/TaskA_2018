from tensorflow.keras.layers import BatchNormalization, \
    Conv2D, Activation, AveragePooling2D, Input, Flatten, Dense, MaxPooling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from utils.utils import *


def project_shortcut_block(input_tensor, filters, stage, block, strides=(2, 2), bn_axis=-1, weight_decay=1e-4):
    """解決唯独匹配的捷径卷积块"""
    conv_name_base = 'res' + str(stage) + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + str(block) + '_branch'

    filter1, filter2, filter3 = filters

    # 1x1 for reducing filters and size of feature maps
    x = Conv2D(filter1,
               (1, 1),
               strides=strides,
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(weight_decay),
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2,
               (3, 3),
               padding='same',
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(weight_decay),
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 1x1 for increase/restore filters
    x = Conv2D(filter3,
               (1, 1),
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(weight_decay),
               name=conv_name_base + '2c')(x)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # 1x1 for input_tensor to match dimension ,just only make filters equal to filter3,and reduce size of feature maps
    short_cut = Conv2D(filter3,
                       (1, 1),
                       strides=strides,
                       kernel_initializer='he_uniform',
                       kernel_regularizer=l2(weight_decay),
                       name=conv_name_base + '1')(input_tensor)
    short_cut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(short_cut)

    # shortcut in mismatch dimension
    x = add([x, short_cut])  # channels unchanged ,channels ==filter3
    x = Activation('relu')(x)

    return x


def identity_shortcut_block(input_tensor, filters, stage, block, bn_axis=-1, weight_decay=1e-4):
    """ 恒等捷径卷积块"""
    conv_name_base = 'res' + str(stage) + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + str(block) + '_branch'

    filter1, filter2, filter3 = filters

    # 1x1 for reduce filters
    x = Conv2D(filter1,
               (1, 1),
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(weight_decay),
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2,
               (3, 3),
               padding='same',
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(weight_decay),
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 1x1 for increase/restore filters to filter3
    x = Conv2D(filter3,
               (1, 1),
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(weight_decay),
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # shortcut in match dimension
    x = add([x, input_tensor])  # filter doubled after add op
    x = Activation('relu')(x)

    return x


def resNet(input_shape, nums_in_each_part, num_classes, data_format='channels_last'):
    """
    Argument
    :param input_shape:
    :param nums_in_each_part:
    :param num_classes:
    :param data_format:
    :return:
    """
    bn_axis = -1 if data_format == 'channel_last' else 1

    input = Input(input_shape)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    nums1, nums2, nums3, nums4 = nums_in_each_part

    x = project_shortcut_block(x, [64, 64, 256], 1, 0, strides=(1, 1))
    for i in range(1, nums1):
        x = identity_shortcut_block(x, [64, 64, 256], 1, i)

    x = project_shortcut_block(x, [128, 128, 512], 2, 0)
    for i in range(1, nums2):
        x = identity_shortcut_block(x, [128, 128, 512], 2, i)

    x = project_shortcut_block(x, [256, 256, 1024], 3, 0)
    for i in range(1, nums3):
        x = identity_shortcut_block(x, [256, 256, 1024], 3, i)

    x = project_shortcut_block(x, [512, 512, 2048], 4, 0)
    for i in range(1, nums4):
        x = identity_shortcut_block(x, [512, 512, 2048], 4, i)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(input, x, name='resnet')
    describe_model(model)
    return model


def resNet_101(input_shape, num_classes, data_format='channels_last'):
    """
    Argument
    :param input_shape:
    :param num_classes:
    :param data_format:
    :return:
    """

    return resNet(input_shape, nums_in_each_part=[3, 4, 23, 3], num_classes=num_classes, data_format=data_format)


def resNet_50(input_shape, num_classes, data_format='channels_last'):
    """
    Argument
    :param input_shape:
    :param num_classes:
    :param data_format:
    :return:
    """

    return resNet(input_shape, nums_in_each_part=[3, 4, 6, 3], num_classes=num_classes, data_format=data_format)


def resNet_152(input_shape, num_classes, data_format='channels_last'):
    """
    Argument
    :param input_shape:
    :param num_classes:
    :param data_format:
    :return:
    """

    return resNet(input_shape, nums_in_each_part=[3, 8, 36, 3], num_classes=num_classes, data_format=data_format)


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
    model = resNet_152(input_shape=(224, 224, 3), num_classes=1000)

    describe_model(model)
