import tensorflow as tf
import numpy as np

def conv_factory(x, filter_size, kernel_size, conv_strides, is_train, pure = False):
    conv = tf.layers.conv2d(x, filters=filter_size, kernel_size=kernel_size,
                          strides=[conv_strides, conv_strides], padding='SAME', activation=None)
    if pure:
        return conv
    else:
        bn = tf.layers.batch_normalization(conv, training=is_train)
        relu = tf.nn.relu(bn)
        return relu

def deconv_factory(x, filter_size, kernel_size, conv_strides, is_train, pure = False):
    deconv = tf.layers.conv2d_transpose(x, filters=filter_size, kernel_size=kernel_size,
                          strides=[conv_strides, conv_strides], padding='SAME', activation=None)
    if pure:
        return deconv
    else:
        bn = tf.layers.batch_normalization(deconv, training=is_train)
        relu = tf.nn.relu(bn)
        return relu

def mobile_conv_factory(x, filter_size, kernel_size, conv_strides, is_train):
    # depthwise + pointwise conv
    conv = tf.layers.separable_conv2d(x, filters=filter_size, kernel_size=kernel_size,
                                  strides=[conv_strides, conv_strides], padding='SAME', activation=None)
    bn = tf.layers.batch_normalization(conv, training=is_train)
    relu = tf.nn.relu(bn)
    return relu

def residual_block_factory(x, filter_size, kernel_size, conv_strides, is_train):
    # 1x1 + (3x3)->(3x3)
    identity = tf.layers.conv2d(x, filters=filter_size, kernel_size=1,
                                strides=[conv_strides, conv_strides], padding='SAME', activation=None)
    identity = tf.layers.batch_normalization(identity, training=is_train)

    residual = tf.layers.conv2d(x, filters=filter_size, kernel_size=kernel_size,
                                strides=[conv_strides, conv_strides], padding='SAME', activation=None)
    residual = tf.layers.batch_normalization(residual, training=is_train)
    residual = tf.nn.relu(residual)
    residual = tf.layers.conv2d(residual, filters=filter_size, kernel_size=kernel_size,
                                strides=[conv_strides, conv_strides], padding='SAME', activation=None)
    residual = tf.layers.batch_normalization(residual, training=is_train)

    return tf.nn.relu(residual+identity)

def encoder(x, filter_size, kernel_size, conv_strides, is_train, hierarchy):
    # downsample by strided conv, another way is pooling
    assert conv_strides > 1
    for i in range(hierarchy):
        x = conv_factory(x, filter_size, kernel_size, conv_strides, is_train)
        filter_size *= 2
    return x

def decoder(x, filter_size, output_size, kernel_size, conv_strides, is_train, hierarchy):
    # upsampling by strided transpose conv, another way is unpooling
    assert conv_strides > 1
    for i in range(hierarchy-1):
        x = deconv_factory(x, filter_size, kernel_size, conv_strides, is_train)
        filter_size = int(filter_size / 2)
    x = deconv_factory(x, output_size, kernel_size, conv_strides, is_train, pure=True)
    return x
