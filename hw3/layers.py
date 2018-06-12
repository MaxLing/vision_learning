import tensorflow as tf
import numpy as np

def conv_factory(x, filter_size, kernel_size, conv_strides, pool_strides, is_train):
    conv = tf.layers.conv2d(x, filters=filter_size, kernel_size=kernel_size,
                          strides=[conv_strides, conv_strides], padding='SAME', activation=None)
    bn = tf.layers.batch_normalization(conv, training=is_train)
    relu = tf.nn.relu(bn)
    pool = tf.layers.max_pooling2d(relu, pool_size=[pool_strides, pool_strides], strides=pool_strides, padding='valid')
    return pool

def mobile_conv_factory(x, filter_size, kernel_size, conv_strides, pool_strides, is_train):
    # depthwise + pointwise conv
    conv = tf.layers.separable_conv2d(x, filters=filter_size, kernel_size=kernel_size,
                                  strides=[conv_strides, conv_strides], padding='SAME', activation=None)
    bn = tf.layers.batch_normalization(conv, training=is_train)
    relu = tf.nn.relu(bn)
    pool = tf.layers.max_pooling2d(relu, pool_size=[pool_strides, pool_strides], strides=pool_strides, padding='valid')
    return pool

def residual_block_factory(x, filter_size, kernel_size, conv_strides, pool_strides, is_train):
    # 1x1 + (3x3)->(3x3)
    identity = tf.layers.conv2d(x, filters=filter_size, kernel_size=1,
                                strides=[conv_strides, conv_strides], padding='SAME', activation=None)
    identity = tf.layers.batch_normalization(identity, training=is_train)

    residual = tf.layers.conv2d(x, filters=filter_size, kernel_size=3,
                                strides=[conv_strides, conv_strides], padding='SAME', activation=None)
    residual = tf.layers.batch_normalization(residual, training=is_train)
    residual = tf.nn.relu(residual)
    residual = tf.layers.conv2d(residual, filters=filter_size, kernel_size=3,
                                strides=[conv_strides, conv_strides], padding='SAME', activation=None)
    residual = tf.layers.batch_normalization(residual, training=is_train)

    return tf.nn.relu(residual+identity)
