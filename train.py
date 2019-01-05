from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import time
import logging
import pickle

from _functions import *
from model import *
from test import *

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
  train_Gen, train_Dis, G_loss, D_loss, loss, Decoded, monitor = model(input, [target15, target14, target13, target12, target11, target10], training, alpha)

  saver = tf.train.Saver(max_to_keep=4)
  print('total number of parameters:', total_parameters())

new_model = True
batch_size = 8
logging.basicConfig(filename='./output/train.log',level=logging.INFO)
with open('train_names.pkl', 'rb') as f:
    train_names = pickle.load(f)
with open('test_names.pkl', 'rb') as f:
    test_names = pickle.load(f)
test_names = np.array(test_names)

print("Training begins")
with tf.Session(graph=graph) as sess:
  if new_model:
    sess.run(tf.global_variables_initializer())
    set_pretrained(sess)
    global_step = 0
    EMA = 0
    train_MAEs = None
  else:
    saver.restore(sess, tf.train.latest_checkpoint('./model/'))
#     saver.restore(sess, './model-ccmv2sha-12312249')
    global_step = 7418
    EMA = 0
    train_MAEs = None

  for step in range(100000):

    train_inputs, train_targets = next_batch(batch_size, train_names)
    train_t15, train_t14, train_t13, train_t12, train_t11, train_t10 = train_targets
    _ , train_loss_G, train_loss = sess.run([train_Gen, G_loss, loss], feed_dict={
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
    _ , train_loss_D,  = sess.run([train_Dis, D_loss], feed_dict={
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
        EMA = train_loss_G
    else:
        EMA = moving_average(train_loss_G, EMA)
    if step%100==0:

      train_D = sess.run(Decoded, feed_dict={
          input: train_inputs,
          target15: train_t15,
          target14: train_t14,
          target13: train_t13,
          target12: train_t12,
          target11: train_t11,
          target10: train_t10,
          training: True,
      })
      train_out15, train_out14, train_out13, train_out12, train_out11, train_out10 = train_D

      if train_MAEs is None:
        train_MAEs = [ MAE(train_out15, train_t15), MAE(train_out14, train_t14), MAE(train_out13, train_t13), MAE(train_out12, train_t12), MAE(train_out11, train_t11), MAE(train_out10, train_t10)]
      else:
        train_MAEs = moving_average_array([ MAE(train_out15, train_t15), MAE(train_out14, train_t14), MAE(train_out13, train_t13),
                MAE(train_out12, train_t12), MAE(train_out11, train_t11), MAE(train_out10, train_t10)], train_MAEs)

      log_str = ['>>> TRAIN', time.asctime()[10:20]+': i [', str(global_step), '] || [loss, EMA]: [',
                   str((round(train_loss, 2), round(train_loss_G-train_loss, 2), round(train_loss_D, 2)))+',', str(round(EMA,2)), '] || [EMAoMAE]:', str(train_MAEs)]
      print(*log_str)
      logging.info(' '.join(log_str))

      if step%1000==0 and False:

        display_set_of_imgs([train_out14[0], train_t14[0], train_out13[0], train_t13[0], train_out12[0]
                             , train_t12[0], train_out11[0], train_t11[0], train_out10[0], train_t10[0], denormalize(train_inputs[0])]
                             , rows=3, size=2, name='train-'+str(global_step))

      if step%1000==0:

        saver.save(sess, "./model/model", global_step=global_step)
        print(">>> Model saved")
        logging.info(">>> Model saved")

    if (step % 1000==0 and step>=3000) or step==0:
        test_results = full_test(sess, Decoded,
            input, target15, target14, target13, target12, target11, target10, training)
        log_str = ['>>> TEST ', time.asctime()+': i [', str(global_step), '] || [Result]:', str(test_results)]
        print(*log_str)
        logging.info(' '.join(log_str))
        
    global_step = global_step + 1
