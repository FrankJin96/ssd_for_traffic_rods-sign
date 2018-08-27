# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 08:51:20 2017

@author: Frank_Jin
"""

# In[]
'''

voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
NUM_CLASSES = len(voc_classes) + 1

'''
# In[]

from keras import Sequential
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D

model = Sequential()

# Block 1
model.add(Conv2D(64, (3, 3),
                 name='conv1_1',
                 padding='same',
                 activation='relu',
                 input_shape=(300,300,3)))
first_layer = model.layers[-1]
# this is a placeholder tensor that will contain our generated images
input_img = first_layer.input

# build the rest of the network
model.add(Conv2D(64, (3, 3),
                 name='conv1_2',
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(name='pool1',
                       pool_size=(2, 2),
                       strides=(2, 2),
                       padding='same', ))

# Block 2
model.add(Conv2D(128, (3, 3),
                 name='conv2_1',
                 padding='same',
                 activation='relu'))
model.add(Conv2D(128, (3, 3),
                 name='conv2_2',
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(name='pool2',
                       pool_size=(2, 2),
                       strides=(2, 2),
                       padding='same'))

# Block 3
model.add(Conv2D(256, (3, 3),
                 name='conv3_1',
                 padding='same',
                 activation='relu'))
model.add(Conv2D(256, (3, 3),
                 name='conv3_2',
                 padding='same',
                 activation='relu'))
model.add(Conv2D(256, (3, 3),
                 name='conv3_3',
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(name='pool3',
                       pool_size=(2, 2),
                       strides=(2, 2),
                       padding='same'))

# Block 4
model.add(Conv2D(512, (3, 3),
                 name='conv4_1',
                 padding='same',
                 activation='relu'))
model.add(Conv2D(512, (3, 3),
                 name='conv4_2',
                 padding='same',
                 activation='relu'))
model.add(Conv2D(512, (3, 3),
                 name='conv4_3',
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(name='pool4',
                       pool_size=(2, 2),
                       strides=(2, 2),
                       padding='same'))

# Block 5
model.add(Conv2D(512, (3, 3),
                 name='conv5_1',
                 padding='same',
                 activation='relu'))
model.add(Conv2D(512, (3, 3),
                 name='conv5_2',
                 padding='same',
                 activation='relu'))
model.add(Conv2D(512, (3, 3),
                 name='conv5_3',
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(name='pool5',
                       pool_size=(3, 3),
                       strides=(1, 1),
                       padding='same'))

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])


# In[]

weights_path = 'D:/Documents/ssd_keras-master/SSD_KERAS/ssd_keras_v2/weights.30-1.22.hdf5'

'''
import h5py

f = h5py.File(weights_path,'r')
for k in range(f.attrs['nb_layers']):
    ###############################################
    #####这里的f.attrs[]不懂，nb_layers也没找到#####
    ##############下面的nb_params亦是##############
    ###############################################
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
# 在ipython试试f[k].values能不能用set_weights来装载
'''

model.load_weights(weights_path, by_name = True)
print('Model loaded.')


# In[]

from keras import backend as K

layer_name = 'conv5_1'
filter_index = 1  # can be any integer from 0 to 511, as there are 512 filters in that layer

# build a loss function that maximizes the activation
# of the nth filter of the layer considered
layer_output = layer_dict[layer_name].output
#############################################
###########dict的output是什么操作?############
#############################################
loss = K.mean(layer_output[:, filter_index, :, :])   #####K.mean返回当前张量的均值#####

# compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# this function returns the loss and grads given the input picture
iterate = K.function([input_img], [loss, grads])    #####K.function使得形成一个函数#####


# In[]

import numpy as np

step = 0.1
img_width, img_height = 300, 300
# we start from a gray image with some noise
input_img_data = np.random.random((1, img_width, img_height, 3)) * 20 + 128.
# run gradient ascent for 20 steps
for i in range(20):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step     #####step应该是多少#####


# In[]
from scipy.misc import imsave

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()             #####mean的用法####
    x /= (x.std() + 1e-5)     #####std的用法#####
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)     #####clip使得超出min/max的变为min/max#####

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

img = input_img_data[0]
img = deprocess_image(img)
sum_img = img[:,0,:] + img[:,1,:] + img[:,2,:]
#imsave('%s_filter_%d.png' % (layer_name, filter_index), img)
imsave('vzl.png', sum_img)
