# -*- coding: utf-8 -*-
"""
Created on 2022.5.25 11:56
Location jianke

@author: Troye Jcan
"""
import keras.backend as K
from keras import layers
from keras.applications import imagenet_utils
from keras.models import Model
from keras.utils import layer_utils

seed = 42


def get_input_shape(_shape, _tensor, _only_conv):
    input_shape = imagenet_utils.obtain_input_shape(
        _shape,
        default_size=224,
        min_size=32,
        data_format=K.image_data_format(),
        require_flatten=_only_conv)

    if _tensor is None:
        inputs = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(_tensor):
            inputs = layers.Input(tensor=_tensor, shape=input_shape)
        else:
            inputs = _tensor
    return inputs


def VGG(cfg='vgg16',
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation='softmax',
        if_dropout=False,
        dropout_rate=0.5,
        only_conv=False):
    """
    VGG model

    https://arxiv.org/pdf/1409.1556.pdf

    :param cfg: vgg model configuration, default is vgg16
    :param input_shape: default is None, if None, the input shape is (224, 224, 3)
    :param input_tensor: Optional Keras tensor, to use as the input for the model.
    :param num_classes: number of classes
    :param classifier_activation: activation function for class prediction
    :param if_dropout: if use dropout
    :param dropout_rate: if use dropout, dropout rate
    :param only_conv: whether to include the 3 fully-connected layers.
    :return: vgg model
    """
    cfgs = {
        'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }
    assert cfg in cfgs.keys(), "not support model {}, only {} is supported".format(cfg, cfgs.keys())
    layer_list = cfgs[cfg]

    inputs = get_input_shape(input_shape, input_tensor, only_conv)

    x = inputs
    block_num, conv_num = 1, 1
    for filters in layer_list:
        if filters == 'M':
            x = layers.MaxPool2D()(x)
            block_num += 1
            conv_num = 1
        else:
            x = layers.Conv2D(filters, kernel_size=3, padding='same',
                              activation='relu', name=f"block{block_num}_conv{conv_num}")(x)
            conv_num += 1
    if not only_conv:
        x = layers.Flatten()(x)
        if if_dropout:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(2048, activation='relu', name='fc1')(x)
        if if_dropout:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(2048, activation='relu', name='fc2')(x)
        x = layers.Dense(num_classes, activation=classifier_activation, name='predictions')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    model = Model(inputs=inputs, outputs=x, name=cfg)
    return model


def conv_block(x, kernel, filters, conv_num, strides=1, conv_shortcut=True, name=None):
    if conv_num == 2:
        redusial = layers.Conv2D(filters, 1, strides=strides, name=name + '_0_conv')(x)
        redusial = layers.BatchNormalization(epsilon=1.001e-5, name=name + '_0_bn')(redusial)
    elif conv_shortcut:
        redusial = layers.Conv2D(4 * filters, 1, strides=strides, name=name + '_0_conv')(x)
        redusial = layers.BatchNormalization(epsilon=1.001e-5, name=name + '_0_bn')(redusial)
    else:
        redusial = x
    x = layers.Conv2D(filters, kernel_size=kernel[0], strides=strides, padding='same', name=name + "_conv1")(x)
    x = layers.BatchNormalization(name=name+'_bn1')(x)
    x = layers.Activation('relu', name=name+'_relu1')(x)
    x = layers.Conv2D(filters, kernel_size=kernel[1], padding='same', name=name + "_conv2")(x)
    x = layers.BatchNormalization(name=name+'_bn2')(x)
    x = layers.Activation('relu', name=name+'_relu2')(x)
    if conv_num == 3:
        x = layers.Conv2D(filters * 4, kernel_size=kernel[2], padding='same', name=name + "_conv3")(x)
        x = layers.BatchNormalization(name=name+'_bn3')(x)
        x = layers.Activation('relu', name=name+'_relu3')(x)
    x = layers.Add(name=name+'_add')([redusial, x])
    x = layers.Activation('relu', name=name+'_out')(x)
    return x


def ResNet(cfg='resnet50',
           input_shape=None,
           input_tensor=None,
           num_classes=1000,
           classifier_activation='softmax',
           if_dropout=False,
           dropout_rate=0.5,
           only_conv=False,
           if_v2=False):
    """
    ResNet model

    https://arxiv.org/pdf/1512.03385.pdf

    :param cfg:
    :param input_shape:
    :param input_tensor:
    :param num_classes:
    :param classifier_activation:
    :param if_dropout:
    :param dropout_rate:
    :param only_conv:
    :param if_v2:
    :return:
    """
    cfgs = {
        'resnet18': [[64, 128, 256, 512], [2, 2, 2, 2], [3, 3]],
        'resnet34': [[64, 128, 256, 512], [3, 4, 6, 3], [3, 3]],
        'resnet50': [[64, 128, 256, 512], [3, 4, 6, 3], [1, 3, 1]],
        'resnet101': [[64, 128, 256, 512], [3, 4, 23, 3], [1, 3, 1]],
        'resnet152': [[64, 128, 256, 512], [3, 8, 36, 3], [1, 3, 1]],
    }
    assert cfg in cfgs.keys(), "not support model {}, only {} is supported".format(cfg, cfgs.keys())
    layer_list = cfgs[cfg]

    inputs = get_input_shape(input_shape, input_tensor, only_conv)
    x = layers.ZeroPadding2D(padding=(3, 3), name='padding0')(inputs)
    x = layers.Conv2D(64, kernel_size=7, strides=2, name='conv0')(x)

    if not if_v2:
        x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='conv0_bn')(x)
        x = layers.Activation('relu', name='conv0_relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='padding1')(x)
    x = layers.MaxPooling2D(3, strides=2, name='maxpool0')(x)
    for filters in range(len(layer_list[0])):
        strides = 1 if filters == 0 else 2
        kernel_list = layer_list[2]
        conv_filter = layer_list[0][filters]
        x = conv_block(x, kernel_list, conv_filter, len(kernel_list), strides, name=f"conv{filters + 2}_block1")
        for block in range(2, layer_list[1][filters]+1):
            x = conv_block(x, kernel_list, conv_filter, len(kernel_list), conv_shortcut=False,
                           name='conv{}_block{}'.format(filters+2, block + 1))

    if if_v2:
        x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)

    if not only_conv:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if if_dropout:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(num_classes, activation=classifier_activation, name='predictions')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    model = Model(inputs=inputs, outputs=x, name=cfg)
    return model


if __name__ == '__main__':
    _input_tensor = layers.Input(shape=(224, 224, 3))
    resnet = ResNet('resnet34', input_tensor=_input_tensor, only_conv=False)
    resnet.summary()
