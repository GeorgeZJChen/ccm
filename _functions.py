from __future__ import print_function, division
import torch
from torchvision import models as torch_models
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

def move_files(path_to_load, part='A'):
  if not path_to_load.endswith('/'):
    path_to_load += '/'
  train_ptl = path_to_load + 'train/'
  test_ptl = path_to_load + 'test/'

  if not os.path.exists(train_ptl):
    os.makedirs(train_ptl)
  if not os.path.exists(test_ptl):
    os.makedirs(test_ptl)
  for _, _, files in os.walk("./shanghaitech/part_"+part+"_final/train_data/ground_truth"):
    for filename in files:
      if '.mat' in filename:
        new_name = filename.replace('GT_','')
        os.rename("./shanghaitech/part_"+part+"_final/train_data/ground_truth/"+filename, train_ptl + new_name)
        os.rename("./shanghaitech/part_"+part+"_final/train_data/images/"+new_name.replace('.mat','.jpg'), train_ptl + new_name.replace('.mat','.jpg'))
  for _, _, files in os.walk("./shanghaitech/part_"+part+"_final/test_data/ground_truth"):
    for filename in files:
      if '.mat' in filename:
        new_name = filename.replace('GT_','')
        os.rename("./shanghaitech/part_"+part+"_final/test_data/ground_truth/"+filename, test_ptl + new_name)
        os.rename("./shanghaitech/part_"+part+"_final/test_data/images/"+new_name.replace('.mat','.jpg'), test_ptl + new_name.replace('.mat','.jpg'))

def load_data_names(train=True, part='A'):
  names = []
  if train:
    for _, _, files in os.walk('./datasets/shanghaitech/'+part+'/train/'):
      for filename in files:
        if '.mat' in filename:
          names.append(filename.replace('.mat',''))
  else:
    pass
    for _, _, files in os.walk('./datasets/shanghaitech/'+part+'/test'):
        for filename in files:
          if '.jpg' in filename:
            names.append(filename.replace('.jpg',''))
  return names
def load_data_ShanghaiTech(path):
  img = Image.open(path+'.jpg')
  coords = scipy_io.loadmat(path+'.mat')['image_info'][0][0][0][0][0]
  return img, coords
def display_set_of_imgs(images, rows=2, size=0.5, name='0'):
  n_images = len(images)
  with open('./output/images/'+str(name)+'-'+id_generator(5)+'.pkl', 'wb') as f:
      pickle.dump(images, f)

  # fig = plt.figure()
  # plt.axis('off')
  # for n, image in enumerate(images):
  #     if image.shape[-1] == 1:
  #       image = np.reshape(image, (image.shape[0], image.shape[1]))
  #       a = fig.add_subplot(rows, np.ceil(n_images/float(rows)), n + 1)
  #       a.axis('off')
  #       a.set_title(str(image.shape)+', '+str(round(np.sum(image), 2)))
  #       plt.imshow(image, cmap=plt.get_cmap('jet'))
  #     elif  image.shape[-1] == 3:
  #       a = fig.add_subplot(rows, np.ceil(n_images/float(rows)), n + 1)
  #       a.axis('off')
  #       a.set_title(str(image.shape))
  #       plt.imshow(image)
  # fig.set_size_inches(np.array(fig.get_size_inches()) * size)
  # plt.show()
def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
  return ''.join(random.choice(chars) for _ in range(size))
def total_parameters(scope=None):
  total_parameters = 0
  for variable in tf.trainable_variables(scope):
      # shape is an array of tf.Dimension
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
          variable_parameters *= dim.value
      total_parameters += variable_parameters
  return total_parameters
def gaussian_kernel(shape=(32,32),sigma=5):
  """
  2D gaussian kernel which is equal to MATLAB's
  fspecial('gaussian',[shape],[sigma])
  """
  radius_x,radius_y = [(radius-1.)/2. for radius in shape]
  y_range,x_range = np.ogrid[-radius_y:radius_y+1,-radius_x:radius_x+1]
  h = np.exp( -(x_range*x_range + y_range*y_range) / (2.*sigma*sigma) )

  # finfo(dtype).eps: a very small value
  h[ h < np.finfo(h.dtype).eps*h.max()] = 0
  sumofh = h.sum()
  if sumofh != 0:
      h /= sumofh
  return h
def get_downsized_density_maps(density_map):
  ddmaps = []
  ratios = [8,16,32,64,128]
  with tf.device('/gpu:0'):
    ddmap = tf.layers.average_pooling2d(density_map, ratios[0], ratios[0], padding='same') * (ratios[0] * ratios[0])
    ddmaps.append(tf.squeeze(ddmap,0))
    if len(ratios)>1:
      for i in range(len(ratios)-1):
        ratio = int(ratios[i+1]/ratios[i])
        ddmap = tf.layers.average_pooling2d(ddmap, ratio, ratio, padding='same') * (ratio * ratio)
        ddmaps.append(tf.squeeze(ddmap,0))

  return ddmaps, [tf.image.flip_left_right(ddmap) for ddmap in ddmaps]
def random_size(rate_range=[1.1, 1.6], input_size=[384, 512], img_size=[None,None]):
  img_height, img_width = img_size
  input_height, input_width = input_size
  resized_height = img_height
  resized_width = img_width
  erate = rate_range[0] + (rate_range[1]-rate_range[0])*random.random()
  if img_height <= input_height*erate:
    resized_height = int(input_height*erate)
    resized_width = resized_height/img_height*img_width
    if resized_width <= input_width*erate:
      resized_width = int(input_width*erate)
      resized_height = resized_width/img_width*img_height
  elif  img_width <= input_width*erate:
    resized_width = int(input_width*erate)
    resized_height = resized_width/img_width*img_height
    if resized_height <= input_height*erate:
      resized_height = int(input_height*erate)
      resized_width = resized_height/img_height*img_width
  return int(resized_height), int(resized_width)
def fit_grid(img_height, img_width, input_size=[384,512]):
  input_height, input_width = input_size
  columns = max(1, int(round(img_width/input_width)))
  rows = max(1, int(round(input_width*columns*img_height/img_width/input_height)))
  return rows, columns
def get_coords_map(coords, resize, img_size):
  resized_height, resized_width = resize
  img_height, img_width = img_size
  new_coords = []
  for coord in coords:
    new_coord = [0,0]
    new_coord[0] = min(coord[0], img_width-1)*resized_width/img_width
    new_coord[1] = min(coord[1], img_height-1)*resized_height/img_height
    new_coords.append(new_coord)
  coords_map = np.zeros([1, resized_height, resized_width, 1])
  for coord in new_coords:
    coords_map[0][int(coord[1])][int(coord[0])][0] += 1
  return coords_map
def get_resized_image_and_density_map(img, coords_map, kernel, resize):
  img_shape = img.get_shape().as_list()
  img_height, img_width = img_shape[1], img_shape[2]
  resized_height, resized_width = resize

  with tf.device('/gpu:0'):
    coords_map = tf.constant(coords_map, dtype=tf.float32)
    density_map = tf.nn.conv2d(coords_map, kernel, strides=(1,1,1,1), padding='SAME')

  img_height, img_width = img.shape[0], img.shape[1]
  img = tf.image.resize_images(img, [resized_height, resized_width])
  img = tf.cast(img, tf.uint8)
  return img, density_map

def preprocess_data(names, data_path, save_path='./processed', whole_image=True, random_crop=None, quarter_crops=True, input_size=[384, 512]):
  if not data_path.endswith('/'):
    data_path += '/'
  if not save_path.endswith('/'):
    save_path += '/'
  if not os.path.exists(save_path):
    os.makedirs(save_path)

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

    if whole_image:

      resized_height = input_height
      resized_width = input_width

      new_img = img.resize((resized_width, resized_height))
      coords_map = get_coords_map(coords, resize=[resized_height, resized_width], img_size=[img_height, img_width])

      dmap = sess_get_dmap.run(tf_dmap, feed_dict={
          tf_coords_map_p: coords_map
      })
      ddmaps, ddmaps_mirrored = sess_get_downsized_dmaps.run(tf_ddmaps, feed_dict={
          tf_dmap_p: dmap
      })

      imgs.append(new_img)
      dmaps.append(ddmaps)

      imgs.append(ImageOps.mirror(new_img))
      dmaps.append(ddmaps_mirrored)

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

          imgs.append(ImageOps.mirror(img_crop))
          dmaps.append(ddmaps_mirrored)

    if random_crop is not None:
      for b in range(random_crop):

        resized_height, resized_width = random_size(input_size=[input_height, input_width], img_size=(img_height, img_width))
        new_img = img.resize((resized_width, resized_height))
        coords_map = get_coords_map(coords, resize=[resized_height, resized_width], img_size=[img_height, img_width])

        dmap = sess_get_dmap.run(tf_dmap, feed_dict={
            tf_coords_map_p: coords_map
        })

        crop_top = np.random.randint(0, resized_height - input_height)
        crop_left = np.random.randint(0, resized_width - input_width)
        crop_bottom = crop_top + input_height
        crop_right = crop_left + input_width
        img_crop = new_img.crop((crop_left, crop_top, crop_right, crop_bottom))

        ddmaps, ddmaps_mirrored = sess_get_downsized_dmaps.run(tf_ddmaps, feed_dict={
            tf_dmap_p: dmap[:, crop_top:crop_bottom, crop_left:crop_right]
        })

        imgs.append(img_crop)
        dmaps.append(ddmaps)

        imgs.append(ImageOps.mirror(img_crop))
        dmaps.append(ddmaps_mirrored)

    for i in range(len(imgs)):
      new_name = id_generator()

      imgs[i].save(save_path + new_name + '.jpg', 'JPEG')
      with open(save_path + new_name + '.pkl', 'wb') as f:
        pickle.dump(dmaps[i], f)

      out_names.append(save_path + new_name)
  return out_names

def set_pretrained(sess):

  vgg16 = torch_models.vgg16(pretrained=True)
  torch_dict = vgg16.state_dict()

  tf_p_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  torch_p_ids = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21]
  trainables = tf.trainable_variables()
  for i in range(10):
    tf_name_w = 'generator/vgg_conv_'+str(tf_p_ids[i])+'/kernel:0'
    tf_name_b = 'generator/vgg_conv_'+str(tf_p_ids[i])+'/bias:0'

    torch_name_w = 'features.'+str(torch_p_ids[i])+'.weight'
    torch_name_b = 'features.'+str(torch_p_ids[i])+'.bias'

    var_w = [v for v in trainables if v.name == tf_name_w ][0]
    sess.run(tf.assign(var_w, np.transpose(torch_dict[torch_name_w].data.numpy(), (2,3,1,0))))

    var_b = [v for v in trainables if v.name == tf_name_b ][0]
    sess.run(tf.assign(var_b, torch_dict[torch_name_b].data.numpy()))
#   test_set_pretrained('CAC/vgg_conv_10/kernel:0', 'features.21.weight', torch_dict)

def test_set_pretrained(tf_name, torch_name, torch_dict):
  def check_equal4d(a, b):
    for m in range(a.shape[0]):
      for n in range(a.shape[1]):
        for h in range(a.shape[2]):
          for w in range(a.shape[3]):
            if abs(a[m][n][h][w] - b[m][n][h][w])>0.00001:
              print(a[m][n][h][w], b[m][n][h][w])
              print('at',m,n,h,w)
              return False
    return True
  def check_equal1d(a, b):
    for m in range(a.shape[0]):
      if abs(a[m] - b[m])>0.00001:
        print(a[m], b[m])
        print('at', m)
        return False
    return True
  tf_data = [v for v in tf.trainable_variables() if v.name ==tf_name][0].read_value().eval()
  torch_data = torch_dict[torch_name].data.numpy()
  if len(tf_data.shape) == 1:
    assert check_equal1d(tf_data, torch_data)
  else:
    torch_data = np.transpose(torch_data, (2,3,1,0))
    assert check_equal4d(tf_data, torch_data)

def moving_average(new_val, last_avg, theta=0.95):
  return round((1-theta) * new_val + theta* last_avg, 2)
def moving_average_array(new_vals, last_avgs, theta=0.95):
  return [round((1-theta) * new_vals[i] + theta* last_avgs[i], 2) for i in range(len(new_vals))]
def MAE(predicts, targets):
  return round( np.mean( np.absolute( np.sum(predicts, (1,2,3)) - np.sum(targets, (1,2,3)) )), 1)
def normalize(imgs):
  new_imgs = []
  for i in range(len(imgs)):
    img = imgs[i] / 255
    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]
    new_imgs.append(img)
  return new_imgs
def denormalize(img):
  img *= [0.229, 0.224, 0.225]
  img += [0.485, 0.456, 0.406]
  img *= 255
  return img.astype('uint8')
epoch = 0
batch_step = 0
def next_batch(batch_size, names, train=True):
  imgs = []
  targets15 = []
  targets14 = []
  targets13 = []
  targets12 = []
  targets11 = []
  targets10 = []
  global batch_step
  global epoch
  cb = batch_step
  if batch_step>=len(names):
    batch_step = 0
  if cb+batch_size > len(names):
    batch_step = cb + batch_size - len(names)
    _names = names[cb : len(names)] + names[0: batch_step ]
    random.shuffle(names)
    if train:
      epoch += 1
      print(time.asctime()+':  ', 'epoch', epoch, 'finished')
    else:
      print('________Test epoch finished________')
  else:
    _names = names[cb : cb+batch_size]
    batch_step = cb+batch_size
  for name in _names:
    imgs.append(np.asarray(Image.open(name+'.jpg')))
    target10, target11, target12, target13, target14 = pickle.load(open(name+'.pkl','rb'))
    targets15.append(np.reshape(np.sum(target14), [1,1,1]))
    targets14.append(target14)
    targets13.append(target13)
    targets12.append(target12)
    targets11.append(target11)
    targets10.append(target10)

  targets = [targets15, targets14, targets13, targets12, targets11, targets10]
  return np.array(normalize(imgs)), targets

def next_batch_test(batch_size, names):
  b = np.random.randint(0, len(names), [batch_size])
  _names = names[b]

  imgs = []
  targets15 = []
  targets14 = []
  targets13 = []
  targets12 = []
  targets11 = []
  targets10 = []

  for name in _names:
    imgs.append(np.asarray(Image.open(name+'.jpg')))
    target10, target11, target12, target13, target14 = pickle.load(open(name+'.pkl','rb'))
    targets15.append(np.reshape(np.sum(target14), [1,1,1]))
    targets14.append(target14)
    targets13.append(target13)
    targets12.append(target12)
    targets11.append(target11)
    targets10.append(target10)

  targets = [targets15, targets14, targets13, targets12, targets11, targets10]
  return np.array(normalize(imgs)), targets
