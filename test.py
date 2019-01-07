from __future__ import print_function, division
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import string
import os
import random
import time
from tqdm import tqdm
from scipy import io as scipy_io
import math
import pickle
import h5py
import cv2

from _functions import *
from model import *

def preprocess_data_test(names, data_path, save_path='./processed',
                    quarter_crops=True, input_size=[384, 512],
                    test=False,
                    test_dict=None
                   ):
  if not data_path.endswith('/'):
    data_path += '/'
  if not save_path.endswith('/'):
    save_path += '/'
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  if test and not 'names_to_name' in test_dict:
    test_dict['names_to_name'] = {}

  input_height, input_width = input_size
  prog = 0
  out_names = []
  kernel = gaussian_kernel(shape=(48,48),sigma=10)
  kernel = np.reshape(kernel, kernel.shape+(1,1))

  graph_get_dmap = tf.Graph()
  with graph_get_dmap.as_default():

    kernel = tf.constant(kernel, dtype=tf.float32)

    tf_coords_map_p = tf.placeholder(tf.float32, [1,None,None,1])
    tf_dmap = tf.nn.conv2d(tf_coords_map_p, kernel, strides=(1,1,1,1), padding='SAME')

  graph_get_downsized_dmaps = tf.Graph()
  with graph_get_downsized_dmaps.as_default():

    tf_dmap_p = tf.placeholder(tf.float32, [1,input_height,input_width,1])
    tf_ddmaps = get_downsized_density_maps(tf_dmap_p)

  sess_get_dmap = tf.Session(graph=graph_get_dmap)
  sess_get_downsized_dmaps = tf.Session(graph=graph_get_downsized_dmaps)

  for ni in tqdm(range(len(names))):
    name = data_path +  names[ni]

    img, coords = load_data_ShanghaiTech(name)

    if img.mode !='RGB':
      img = img.convert('RGB')
    img_width, img_height = img.size

    imgs = []
    dmaps = []

    if quarter_crops:
      assert abs(img_height/img_width - input_height/input_width) < 0.2
      resized_height = input_height*2
      resized_width = input_width*2

      new_img = img.resize((resized_width, resized_height))
      coords_map = get_coords_map(coords, resize=[resized_height, resized_width], img_size=[img_height, img_width])
      dmap_crop = sess_get_dmap.run(tf_dmap, feed_dict={
          tf_coords_map_p: coords_map
      })
      for leri in [0,1]:
        for uplo in [0,1]:
          crop_left = input_width*leri
          crop_top = input_height*uplo
          crop_bottom = crop_top + input_height
          crop_right = crop_left + input_width
          img_crop = new_img.crop((crop_left, crop_top, crop_right, crop_bottom))

          ddmaps, ddmaps_mirrored = sess_get_downsized_dmaps.run(tf_ddmaps, feed_dict={
              tf_dmap_p: dmap_crop[:, crop_top:crop_bottom, crop_left:crop_right]
          })
          imgs.append(img_crop)
          dmaps.append(ddmaps)

    for i in range(len(imgs)):
      new_name = id_generator()

      imgs[i].save(save_path + new_name + '.jpg', 'JPEG')
      with open(save_path + new_name + '.pkl', 'wb') as f:
        pickle.dump(dmaps[i], f)

      out_names.append(save_path + new_name)

      if test:
        test_dict['names_to_name'][save_path + new_name] = name
    if test:
      test_dict[name] = {
          'predict': -1
      }

  return out_names

def get_test_names():
    if not (os.path.exists('./test_dict.pkl') and os.path.exists('./strict_test_names.pkl')):
        # tf.reset_default_graph()
        test_dict = {}
        strict_test_names = preprocess_data_test(
            names=load_data_names(train=False, part='A'),
            data_path='./datasets/shanghaitech/A/test/',
            test=True,
            test_dict=test_dict,
            input_size=[384,512]
        )
        random.shuffle(strict_test_names)
        print()
        print(len(strict_test_names), 'of data')
        with open('strict_test_names.pkl', 'wb') as f:
            pickle.dump(strict_test_names, f)
        with open('test_dict.pkl', 'wb') as f:
            pickle.dump(test_dict, f)
    else:
        strict_test_names = pickle.load(open('./strict_test_names.pkl', 'rb'))
        test_dict = pickle.load(open('./test_dict.pkl', 'rb'))
    return strict_test_names, test_dict

def get_data_by_name(name):

  img = np.asarray(Image.open(name+'.jpg'))
  target10, target11, target12, target13, target14 = pickle.load(open(name+'.pkl','rb'))
  target15 = np.reshape(np.sum(target14), [1,1,1])

  targets = [[target15], [target14], [target13], [target12], [target11], [target10]]
  return np.array(normalize([img])), targets


def full_test(sess, Decoded,
    input, target15, target14, target13, target12, target11, target10, training):
  print(">>> Test begins", end='.')
  strict_test_names, test_dict = get_test_names()
  for key in test_dict:
    test_dict[key]['predict'] = np.array([0.0]*6)
    test_dict[key]['truth'] = 0

  step = 0
  for test_name_strict in strict_test_names:

    test_inputs, test_targets = get_data_by_name(test_name_strict)
    test_t15, test_t14, test_t13, test_t12, test_t11, test_t10 = test_targets
    test_D  = sess.run(Decoded, feed_dict={
        input: test_inputs,
        target15: test_t15,
        target14: test_t14,
        target13: test_t13,
        target12: test_t12,
        target11: test_t11,
        target10: test_t10,
        training: False,
    })

    out15, out14, out13, out12, out11, out10 = test_D
    predict = np.sum(out15)
    data_name = test_dict['names_to_name'][test_name_strict]
    test_dict[data_name]['predict'] += np.array([np.sum(out15),np.sum(out14),np.sum(out13),np.sum(out12),np.sum(out11),np.sum(out10)])
    test_dict[data_name]['truth'] += np.sum(test_t15)

    step += 1

    if step % (len(test_names)//15) == 0:
        print('.', end='')
  print()

  results = []
  for key in test_dict:
    if key != 'names_to_name':
      _data = test_dict[key]
      results.append(np.abs(_data['predict']-_data['truth']))

  results = np.mean(results, axis=0)
  return results

def test():
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

    test_names, test_dict = get_test_names()
    for key in test_dict:
      test_dict[key]['predict'] = np.array([0.0]*6)
      test_dict[key]['truth'] = 0

    with tf.Session(graph=graph) as sess:
      saver.restore(sess, tf.train.latest_checkpoint('./'))
      # saver.restore(sess, './model/model-123')
      strict_test_EMA = 0
      step = 0
      for test_name_strict in test_names:

        test_inputs, test_targets = get_data_by_name(test_name_strict)
        test_t15, test_t14, test_t13, test_t12, test_t11, test_t10 = test_targets
        test_D , test_loss,  = sess.run([Decoded, loss], feed_dict={
            input: test_inputs,
            target15: test_t15,
            target14: test_t14,
            target13: test_t13,
            target12: test_t12,
            target11: test_t11,
            target10: test_t10,
            training: False,
        })

        if strict_test_EMA == 0:
          strict_test_EMA = test_loss
        else:
          strict_test_EMA = moving_average(test_loss, strict_test_EMA)
        out15, out14, out13, out12, out11, out10 = test_D
        predict = np.sum(out15)
        data_name = test_dict['names_to_name'][test_name_strict]
        test_dict[data_name]['predict'] += np.array([np.sum(out15),np.sum(out14),np.sum(out13),np.sum(out12),np.sum(out11),np.sum(out10)])
        test_dict[data_name]['truth'] += np.sum(test_t15)

        test_MAEs = [ MAE(out15, test_t15), MAE(out14, test_t14), MAE(out13, test_t13), MAE(out12, test_t12)
                , MAE(out11, test_t11), MAE(out10, test_t10) ]
        MAE_ALL.append(test_MAEs)
        if step%1==0:
          print('>>> ', time.asctime()+': i [', step, '] || [loss, EMA]: [', str(round(test_loss, 3))+',', str(round(strict_test_EMA, 3)), '] || [MAE]:', test_MAEs)
          # if abs(predict-np.sum(test_t15))>15:
          #
          #   display_set_of_imgs([out14[0], test_t14[0], out13[0], test_t13[0], out12[0]
          #                        , test_t12[0], out11[0], test_t11[0], out10[0], test_t10[0], denormalize(test_inputs[0])], rows=3, size=2)
          #   print(np.round(np.mean(np.array(MAE_ALL), 0), 3))
        step += 1

    results = []
    for key in test_dict:
      if key != 'names_to_name':
        _data = test_dict[key]
        results.append(np.abs(_data['predict']-_data['truth']))

    print('>>> test results', np.mean(results, axis=0))

if __name__ == "__main__":
    test()
