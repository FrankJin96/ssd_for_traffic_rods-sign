# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 21:59:42 2018

@author: frank_jin
"""

'''
直接用图片做输入输出，不用考虑什么梯度增强。
tensor.eval()
'''

from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Conv2D, MaxPooling2D
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from imageio import imwrite
import scipy
import os

from keras.applications import inception_v3
# from ssd_v2 import SSD300v2
from keras import backend as K

input_shape = (300, 300, 3)
NUM_CLASSES = 2
weights_path = 'D:/Documents/ssd_keras-master/SSD_KERAS/ssd_keras_v2/weights_SSD300.hdf5'

def preprocess_image(image_path):
    # Util function to open, resize and format pictures
    # into appropriate tensors.
    img = load_img(image_path,target_size=(300, 300))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

def deprocess_image(x):
    # Util function to convert a tensor into a valid image.
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    imwrite(fname, pil_img)
    
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))


inputs = Input(shape=input_shape)

# Block 1
conv1_1 = Conv2D(64, (3, 3),
                     name='conv1_1',
                     padding='same',
                     activation='relu')(inputs)
conv1_2 = Conv2D(64, (3, 3),
                     name='conv1_2',
                     padding='same',
                     activation='relu')(conv1_1)
pool1 = MaxPooling2D(name='pool1',
                     pool_size=(2, 2),
                     strides=(2, 2),
                     padding='same', )(conv1_2)

# Block 2
conv2_1 = Conv2D(128, (3, 3),
                 name='conv2_1',
                 padding='same',
                 activation='relu')(pool1)
conv2_2 = Conv2D(128, (3, 3),
                 name='conv2_2',
                 padding='same',
                 activation='relu')(conv2_1)
pool2 = MaxPooling2D(name='pool2',
                     pool_size=(2, 2),
                     strides=(2, 2),
                     padding='same')(conv2_2)

# Block 3
conv3_1 = Conv2D(256, (3, 3),
                 name='conv3_1',
                 padding='same',
                 activation='relu')(pool2)
conv3_2 = Conv2D(256, (3, 3),
                 name='conv3_2',
                 padding='same',
                 activation='relu')(conv3_1)
conv3_3 = Conv2D(256, (3, 3),
                 name='conv3_3',
                 padding='same',
                 activation='relu')(conv3_2)
pool3 = MaxPooling2D(name='pool3',
                     pool_size=(2, 2),
                     strides=(2, 2),
                     padding='same')(conv3_3)

# Block 4
conv4_1 = Conv2D(512, (3, 3),
                 name='conv4_1',
                 padding='same',
                 activation='relu')(pool3)
conv4_2 = Conv2D(512, (3, 3),
                 name='conv4_2',
                 padding='same',
                 activation='relu')(conv4_1)
conv4_3 = Conv2D(512, (3, 3),
                 name='conv4_3',
                 padding='same',
                 activation='relu')(conv4_2)
pool4 = MaxPooling2D(name='pool4',
                     pool_size=(2, 2),
                     strides=(2, 2),
                     padding='same')(conv4_3)

# Block 5
conv5_1 = Conv2D(512, (3, 3),
                 name='conv5_1',
                 padding='same',
                 activation='relu')(pool4)
conv5_2 = Conv2D(512, (3, 3),
                 name='conv5_2',
                 padding='same',
                 activation='relu')(conv5_1)
conv5_3 = Conv2D(512, (3, 3),
                 name='conv5_3',
                 padding='same',
                 activation='relu')(conv5_2)
pool5 = MaxPooling2D(name='pool5',
                     pool_size=(3, 3),
                     strides=(1, 1),
                     padding='same')(conv5_3)

model = Model(inputs=inputs, outputs=conv1_1)
model.load_weights(weights_path,'r')
model.summary()

pic_name = 'cologne119'
prefix = 'D:/Documents/ssd_keras-master/SSD_KERAS/ssd_keras_v2/target_imgs/'
base_image_path = 'D:/Documents/ssd_keras-master/SSD_KERAS/ssd_keras_v2/target_imgs/' + pic_name + '.png'

img = preprocess_image(base_image_path)
result = model.predict(img, batch_size=1, verbose=1)
# plt.imshow(result[0,:,:,1])

# =============================================================================
# 这段没用
# shp = (result.shape[1], result.shape[2])
# viz_result = result.reshape(2400, 2400)
# save_img(viz_result, fname='456.png') 图片太大，不能保存：tuple index out of range
# cv2.imwrite('456.png', result[0,:,:,1]) 为什么CV2.imwrite写出来的图片全黑？
# =============================================================================
if not os.path.exists(prefix + '{}/'.format(pic_name)):
    os.mkdir(prefix + '{}/'.format(pic_name))
    
for i in range(result.shape[3]):
    Fname = './target_imgs/{}/conv1-1-{:02d}.png'.format(pic_name, (i+1))
    scipy.misc.imsave(Fname, result[0,:,:,i])
    
print('************************')
print('***Saving Images Done***')
print('************************')
