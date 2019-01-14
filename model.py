from __future__ import print_function, division
import tensorflow as tf
import numpy as np

def conv(kernel_size, input, filters, padding='same', strides=(1,1), name=None, act=tf.nn.relu, dilation=1, dropout=None, training=True):
  out = tf.layers.conv2d(input, filters, kernel_size,
                        strides = strides,
                        dilation_rate = 1,
                        activation=act,
                        padding=padding,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                        name=name)
  if dropout is not None:
    out = tf.layers.dropout(out, dropout, training=training)
  return out
def conv_t(kernel_size, input, filters, strides=2, padding='same', act=tf.nn.leaky_relu, dropout=None, training=True):
  out = tf.layers.conv2d_transpose(input, filters, kernel_size, padding=padding, strides=strides, activation=act,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),)
  if dropout is not None:
    out = tf.layers.dropout(out, dropout, training=training)
  return out
def maxpool(kernel_size, input, strides=2):
  return tf.layers.max_pooling2d(input, kernel_size, strides, padding='same')
def abs_loss(predict, target):
  loss = tf.losses.absolute_difference(target, predict, reduction=tf.losses.Reduction.NONE)
  return tf.reduce_mean(loss)
def squared_loss(predict, target):
  loss = tf.losses.mean_squared_error(target, predict, reduction=tf.losses.Reduction.NONE)
  return tf.reduce_mean(loss)
def encoder(input, training, dropout=0.3):
  # input: 384x512
  layer1 = conv(3, input, 64, name='vgg_conv_1')
  layer2 = conv(3, layer1, 64, name='vgg_conv_2')
  pool = maxpool(2, layer2)
  layer3 = conv(3, pool, 128, name='vgg_conv_3')
  layer4 = conv(3, layer3, 128, name='vgg_conv_4')
  pool = maxpool(2, layer4)
  layer5 = conv(3, pool, 256, name='vgg_conv_5')
  layer6 = conv(3, layer5, 256, name='vgg_conv_6')
  layer7 = conv(3, layer6, 256, name='vgg_conv_7') # 96x128 4
  pool = maxpool(2, layer7)
  layer8 = conv(3, pool, 512, name='vgg_conv_8')
  layer9 = conv(3, layer8, 512, name='vgg_conv_9')
  layer10 = conv(3, layer9, 512, name='vgg_conv_10') # 48x64 8

  layer10 = conv(3, layer10, 256, strides=1, dropout=dropout, training=training, act=tf.nn.leaky_relu) # 48x64 8
  print('10', layer10.shape) # 24x32 16

  layer11 = conv(3, layer10, 512, strides=2, dropout=dropout, training=training, act=tf.nn.leaky_relu)
  layer11 = conv(3, layer11, 256, strides=1, dropout=dropout, training=training, act=tf.nn.leaky_relu)
  print('11', layer11.shape) # 24x32 16
  layer12 = conv(3, layer11, 512, strides=2, dropout=dropout, training=training, act=tf.nn.leaky_relu)
  layer12 = conv(3, layer12, 256, strides=1, dropout=dropout, training=training, act=tf.nn.leaky_relu)
  print('12', layer12.shape) # 12x16 32
  layer13 = conv(3, layer12, 512, strides=2, dropout=dropout, training=training, act=tf.nn.leaky_relu)
  layer13 = conv(3, layer13, 256, strides=1, dropout=dropout, training=training, act=tf.nn.leaky_relu)
  print('13', layer13.shape) # 6x8  64
  layer14 = conv(3, layer13, 512, strides=2, dropout=dropout, training=training, act=tf.nn.leaky_relu)
  layer14 = conv(3, layer14, 256, strides=1, dropout=dropout, training=training, act=tf.nn.leaky_relu)
  print('14', layer14.shape) # 3x4  128
  layer15 = conv((3,4), layer14, 1024, padding='valid', act=tf.nn.leaky_relu)
  print('15', layer15.shape) # 1  a

  return layer10, layer11, layer12, layer13, layer14, layer15
def decoder(inputs, training, dropout):
  layer10, layer11, layer12, layer13, layer14, layer15 = inputs

  out15 = conv(1, layer15, 1, act=tf.nn.leaky_relu)
  print('out15', out15.shape)

  layer = conv_t((3,4), layer15, 256, padding='valid', strides=1, dropout=dropout, training=training)
  layer = tf.concat([layer, layer14], axis=3)
  out14 = conv(1, layer, 1, act=tf.nn.leaky_relu)
  print('out14', out14.shape)

  layer = conv_t((3,4), layer15, 256, padding='valid', strides=1, dropout=dropout, training=training)
  layer = tf.concat([layer, layer14], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer13], axis=3)
  out13 = conv(1, layer, 1, act=tf.nn.leaky_relu)
  print('out13', out13.shape)

  layer = conv_t((3,4), layer15, 256, padding='valid', strides=1, dropout=dropout, training=training)
  layer = tf.concat([layer, layer14], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer13], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer12], axis=3)
  out12 = conv(1, layer, 1, act=tf.nn.leaky_relu)
  print('out12', out12.shape)

  layer = conv_t((3,4), layer15, 256, padding='valid', strides=1, dropout=dropout, training=training)
  layer = tf.concat([layer, layer14], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer13], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer12], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer11], axis=3)
  out11 = conv(1, layer, 1, act=tf.nn.leaky_relu)
  print('out11', out11.shape)

  layer = conv_t((3,4), layer15, 256, padding='valid', strides=1, dropout=dropout, training=training)
  layer = tf.concat([layer, layer14], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer13], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer12], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer11], axis=3)
  layer = conv_t(4, layer, 256, dropout=dropout, training=training)
  layer = tf.concat([layer, layer10], axis=3)
  out10 = conv(1, layer, 1, act=tf.nn.leaky_relu)
  print('out10', out10.shape)

  return out15, out14, out13, out12, out11, out10

def model(input, targets, training, alpha, dropout=0.3):

  target15, target14, target13, target12, target11, target10 = targets

  print('input:', input.shape)

  Encoded = encoder(input, training, dropout)
  Decoded = decoder(Encoded, training, dropout)

  out15, out14, out13, out12, out11, out10 = Decoded

  loss = 0
  loss += abs_loss(out15, target15) / 16 / 12 * 10
  loss += abs_loss(out14, target14) / 16 * 2
  loss += abs_loss(out13, target13) / 4
  loss += abs_loss(out12, target12) * 1
  loss += abs_loss(out11, target11) * 4
  loss += abs_loss(out10, target10) * 16

  L2_loss = 0 # tf.losses.get_regularization_loss() * 1e-7

  loss += L2_loss

  trainables = tf.trainable_variables()

  train_vgg = tf.train.MomentumOptimizer(alpha/10, 0.9).minimize(loss, var_list=[var for var in trainables if 'vgg' in var.name])
  train_others = tf.train.MomentumOptimizer(alpha, 0.9).minimize(loss,
                  var_list=[var for var in trainables if 'vgg' not in var.name])
  train = tf.group(train_vgg, train_others)

  D = [tf.nn.relu(out) for out in Decoded]

  m = L2_loss
  return train, loss, D, m
