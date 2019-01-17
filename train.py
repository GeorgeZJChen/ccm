from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import time
import logging
import pickle
import random

from functions import *
from model import *
from test import *
from data import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('resume', nargs='?', default='0')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

tf.reset_default_graph()
print("Initialising Tensors")
# with tf.device('/device:GPU:2'):
if True:
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
    dropout = tf.placeholder_with_default(tf.constant(0.3, tf.float32), shape=[])
    alpha = tf.placeholder_with_default(tf.constant(1e-5, tf.float64), shape=[])
    train, loss, Decoded, monitor = model(input, [target15, target14, target13, target12, target11, target10]
                                                                       , training, alpha, dropout=dropout)
    saver = tf.train.Saver(max_to_keep=2)
    best_saver = tf.train.Saver(max_to_keep=1)
    print('total number of parameters:', total_parameters())

new_model = args.resume!='1'
batch_size = 8
part = 'A'
best_result = 200

logging.basicConfig(filename='./output/train.log',level=logging.INFO)
train_names, test_names = get_train_data_names(part=part)

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
    last_cp = tf.train.latest_checkpoint('./model')
    saver.restore(sess, last_cp)
#     saver.restore(sess, './model-ccmv2sha-01042236')
    global_step = int(last_cp[last_cp.rfind('-')+1:])
    EMA = 0
    train_MAEs = None
    test_MAEs = None

  try:
    for step in range(global_step, 100000):
      if step < 25000:
        lr = 1e-4
      elif step < 50000:
        lr = 1e-5
      elif step < 75000:
        lr = 1e-6
      elif step < 100000:
        lr = 1e-7

      if step%25000==0 and not step==0:
        best_saver.restore(sess, tf.train.latest_checkpoint('./best_model/'))

      train_inputs, train_targets = next_batch(batch_size, train_names)
      train_t15, train_t14, train_t13, train_t12, train_t11, train_t10 = train_targets
      random_dropout = 0 #random.random()*0.3
      _ , train_loss = sess.run([train, loss], feed_dict={
          input: train_inputs,
          target15: train_t15,
          target14: train_t14,
          target13: train_t13,
          target12: train_t12,
          target11: train_t11,
          target10: train_t10,
          training: True,
          alpha: lr,
          dropout: random_dropout,
      })
      if EMA == 0:
        EMA = train_loss
      else:
        EMA = moving_average(train_loss, EMA)
      if step%10==0:

        train_D = sess.run(Decoded, feed_dict={
            input: train_inputs,
            target15: train_t15,
            target14: train_t14,
            target13: train_t13,
            target12: train_t12,
            target11: train_t11,
            target10: train_t10,
            training: True,
            dropout: random_dropout,
        })
        train_out15, train_out14, train_out13, train_out12, train_out11, train_out10 = train_D

        test_inputs, test_targets = next_batch_test(batch_size, test_names)
        test_t15, test_t14, test_t13, test_t12, test_t11, test_t10 = test_targets
        test_D = sess.run(Decoded, feed_dict={
            input: test_inputs,
            target15: test_t15,
            target14: test_t14,
            target13: test_t13,
            target12: test_t12,
            target11: test_t11,
            target10: test_t10,
            training: False,
        })
        test_out15, test_out14, test_out13, test_out12, test_out11, test_out10 = test_D

        if train_MAEs is None:
          train_MAEs = [ MAE(train_out15, train_t15), MAE(train_out14, train_t14), MAE(train_out13, train_t13)
                        , MAE(train_out12, train_t12), MAE(train_out11, train_t11), MAE(train_out10, train_t10)]
        else:
          train_MAEs = moving_average_array([ MAE(train_out15, train_t15), MAE(train_out14, train_t14), MAE(train_out13, train_t13),
                  MAE(train_out12, train_t12), MAE(train_out11, train_t11), MAE(train_out10, train_t10)], train_MAEs)
        if test_MAEs is None:
          test_MAEs = [ MAE(test_out15, test_t15), MAE(test_out14, test_t14), MAE(test_out13, test_t13)
                       , MAE(test_out12, test_t12), MAE(test_out11, test_t11), MAE(test_out10, test_t10)]
        else:
          test_MAEs = moving_average_array([MAE(test_out15,test_t15), MAE(test_out14,test_t14),
                                            MAE(test_out13,test_t13), MAE(test_out12,test_t12),
                  MAE(test_out11,test_t11), MAE(test_out10,test_t10)], test_MAEs)

        log_str = ['>>> TRAIN', time.asctime()[10:20]+': i [', str(global_step), '] || [loss, EMA]: [',
                   str(round(train_loss, 2))+', ', str(round(EMA,2)), '] || [EMAoMAE]:', str(train_MAEs)]
        print(*log_str)
        logging.info(' '.join(log_str))

        log_str = ['>>> TEST ', time.asctime()[10:20]+': i [', str(global_step), '] || [EMAoMAE]:', str(test_MAEs)]
        print(*log_str)
        logging.info(' '.join(log_str))

        if step%400==0 and False:

          display_set_of_imgs([train_out14[0], train_t14[0], train_out13[0], train_t13[0], train_out12[0]
                               , train_t12[0], train_out11[0], train_t11[0], train_out10[0], train_t10[0]
                               , denormalize(train_inputs[0])], rows=3, size=2)
          display_set_of_imgs([test_out14[0], test_t14[0], test_out13[0], test_t13[0], test_out12[0]
                               , test_t12[0], test_out11[0], test_t11[0], test_out10[0], test_t10[0]
                               , denormalize(test_inputs[0])], rows=3, size=2)

        if step%400==0:

          saver.save(sess, "./model/model", global_step=global_step)
          print(">>> Model saved:", global_step)
          logging.info(">>> Model saved: "+str(global_step))

          if global_step>=2000 or step==0:
            test_results = full_test(sess, Decoded,
                input, target15, target14, target13, target12, target11, target10, training, part=part)
            log_str = ['>>> TEST ', time.asctime()+': i [', str(global_step),
                       '] || [Result]:', str([round(result, 2) for result in test_results])]
            if round(test_results[0],2) < best_result:
              best_result = round(test_results[0],2)
              best_saver.save(sess, "./best_model/model-"+str(best_result))
              log_str.append(' * BEST *')
            print(*log_str)
            logging.info(' '.join(log_str))

      global_step = global_step + 1
  except KeyboardInterrupt:
    print('>>> KeyboardInterrupt. Saving model...')
    saver.save(sess, "./model/model", global_step=global_step)
    print(">>> Model saved:", str(global_step))
