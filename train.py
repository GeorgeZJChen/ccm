from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import time
import logging
import pickle

from _functions import *
from model import *

tf.reset_default_graph()
print("Initiating Tensors")
graph = tf.Graph()
with graph.as_default():
  input = tf.placeholder( tf.float32, shape=[None, 384, 512, 3])
  target15 = tf.placeholder( tf.float32 , shape=(None, 1, 1, 1))
  target14 = tf.placeholder( tf.float32 , shape=(None, 3, 4, 1))
  target13 = tf.placeholder( tf.float32 , shape=(None, 6, 8, 1))
  target12 = tf.placeholder( tf.float32 , shape=(None, 12, 16, 1))
  target11 = tf.placeholder( tf.float32 , shape=(None, 24, 32, 1))
  target10 = tf.placeholder( tf.float32 , shape=(None, 48, 64, 1))
  training = tf.placeholder( tf.bool )
  alpha = tf.placeholder_with_default(tf.constant(1e-5, tf.float64), shape=[])
  train, loss, Decoded, monitor = model(input, [target15, target14, target13, target12, target11, target10], training, alpha)

  saver = tf.train.Saver(max_to_keep=4)
  print('total number of parameters:', total_parameters())

new_model = True
batch_size = 16
logging.basicConfig(filename='./output/train.log',level=logging.INFO)
with open('train_names.pkl', 'rb') as f:
    train_names = pickle.load(f)
with open('test_names.pkl', 'rb') as f:
    test_names = pickle.load(f)

print("Training begins")
with tf.Session(graph=graph) as sess:
  if new_model:
    sess.run(tf.global_variables_initializer())
    set_pretrained(sess)
    global_step = 0
    EMA = 0
    train_MAEs = None
    test_MAEs = None
  else:
    saver.restore(sess, tf.train.latest_checkpoint('./model/'))
#     saver.restore(sess, './model-ccmv2sha-12312249')
    global_step = 7418
    EMA = 0
    train_MAEs = None
    test_MAEs = None

  for step in range(100000):

    train_inputs, train_targets = next_batch(batch_size, train_names)
    train_t15, train_t14, train_t13, train_t12, train_t11, train_t10 = train_targets
    _ , train_loss,  = sess.run([train, loss], feed_dict={
        input: train_inputs,
        target15: train_t15,
        target14: train_t14,
        target13: train_t13,
        target12: train_t12,
        target11: train_t11,
        target10: train_t10,
        training: True,
        alpha: 1e-5,
    })
    if EMA == 0:
      EMA = train_loss
    else:
      EMA = moving_average(train_loss, EMA)
    if step%100==0:

      train_D, train_m = sess.run([Decoded, monitor], feed_dict={
          input: train_inputs,
          target15: train_t15,
          target14: train_t14,
          target13: train_t13,
          target12: train_t12,
          target11: train_t11,
          target10: train_t10,
          training: True,
          alpha: 1e-5,
      })
      train_out15, train_out14, train_out13, train_out12, train_out11, train_out10 = train_D

      test_inputs, test_targets = next_batch_test(batch_size, test_names)
      test_t15, test_t14, test_t13, test_t12, test_t11, test_t10 = test_targets
      test_D, test_m = sess.run([Decoded, monitor], feed_dict={
          input: test_inputs,
          target15: test_t15,
          target14: test_t14,
          target13: test_t13,
          target12: test_t12,
          target11: test_t11,
          target10: test_t10,
          training: False,
          alpha: 1e-5,
      })
      test_out15, test_out14, test_out13, test_out12, test_out11, test_out10 = test_D

      if train_MAEs is None:
        train_MAEs = [ MAE(train_out15, train_t15), MAE(train_out14, train_t14), MAE(train_out13, train_t13), MAE(train_out12, train_t12), MAE(train_out11, train_t11), MAE(train_out10, train_t10)]
      else:
        train_MAEs = moving_average_array([ MAE(train_out15, train_t15), MAE(train_out14, train_t14), MAE(train_out13, train_t13),
                MAE(train_out12, train_t12), MAE(train_out11, train_t11), MAE(train_out10, train_t10)], train_MAEs)
      if test_MAEs is None:
        test_MAEs = [ MAE(test_out15, test_t15), MAE(test_out14, test_t14), MAE(test_out13, test_t13), MAE(test_out12, test_t12), MAE(test_out11, test_t11), MAE(test_out10, test_t10)]
      else:
        test_MAEs = moving_average_array([MAE(test_out15,test_t15), MAE(test_out14,test_t14), MAE(test_out13,test_t13), MAE(test_out12,test_t12),
                MAE(test_out11,test_t11), MAE(test_out10,test_t10)], test_MAEs)

      log_str = ['>>> TRAIN', time.asctime()+': i [', str(global_step), '] || [loss, EMA]: [', str(round(train_loss, 3))+',', str(EMA), '] || [EMAoMAE]:', str(train_MAEs)]
      print(*log_str)
      logging.info(' '.join(log_str))

      log_str = ['>>> TEST ', time.asctime()+': i [', str(global_step), '] || [EMAoMAE]:', str(test_MAEs)]
      print(*log_str)
      logging.info(' '.join(log_str))

      if step%1000==0 and True:

        display_set_of_imgs([train_out14[0], train_t14[0], train_out13[0], train_t13[0], train_out12[0]
                             , train_t12[0], train_out11[0], train_t11[0], train_out10[0], train_t10[0], denormalize(train_inputs[0])]
                             , rows=3, size=2, name='train-'+str(global_step))

      if step%1000==0:

        saver.save(sess, "./model/model", global_step=global_step)
        print(">>> Model saved")
        logging.info(">>> Model saved")
    global_step = global_step + 1
