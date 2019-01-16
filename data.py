from __future__ import print_function, division
import tensorflow as tf
import os
import random
import pickle

from functions import *


def get_train_data_names(part):
    if not (os.path.exists('./train_names.pkl') and os.path.exists('./test_names.pkl')):
        move_files('./datasets/shanghaitech/'+part+'/', part = part)
        tf.reset_default_graph()
        train_names = preprocess_data(
            names=load_data_names(train=True, part= part),
            data_path='./datasets/shanghaitech/'+part+'/train/',
            random_crop=30,
            input_size=[384,512]
        )
        random.shuffle(train_names)
        print()
        print(len(train_names), 'of training data')

        test_names = preprocess_data(
            names=load_data_names(train=False, part= part),
            data_path='./datasets/shanghaitech/'+part+'/test/',
            random_crop=5,
            input_size=[384,512]
        )
        random.shuffle(test_names)
        print()
        print(len(test_names), 'of testing data')
        with open('train_names.pkl', 'wb') as f:
            pickle.dump(train_names, f)
        with open('test_names.pkl', 'wb') as f:
            pickle.dump(test_names, f)
    else:
        train_names = pickle.load(open('./train_names.pkl', 'rb'))
        test_names = pickle.load(open('./test_names.pkl', 'rb'))
    return np.array(train_names), np.array(test_names)
