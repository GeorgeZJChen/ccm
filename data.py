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

part = 'B'


move_files('./datasets/shanghaitech/'+part+'/', part = part)

# !rm processed/*
tf.reset_default_graph()
train_names = preprocess_data(
    names=load_data_names(train=True, part= part),
    data_path='./datasets/shanghaitech/'+part+'/train/',
    random_crop=15,
    input_size=[384,512]
)
random.shuffle(train_names)
print()
print(len(train_names), 'of data')

test_names = preprocess_data(
    names=load_data_names(train=True, part= part),
    data_path='./datasets/shanghaitech/'+part+'/test/',
    quarter_crops=True,
    input_size=[384,512]
)
random.shuffle(test_names)
test_names = np.array(test_names)
print()
print(len(test_names), 'of data')

with open('train_names.pkl', 'wb') as f:
    pickle.dump(train_names, f)
with open('test_names.pkl', 'wb') as f:
    pickle.dump(test_names, f)
