# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:41:54 2018

@author: frank jin
"""

'''
To check whether my own dataset is mis-annotated
'''

from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import pickle
from random import shuffle

# In[]

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest' 

np.set_printoptions(suppress=True)

# In[]

gt = pickle.load(open('rod.pkl', 'rb'))

# In[1]:

images = []
img_path = './pics/000001.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
img_path = './pics/000002.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
img_path = './pics/000003.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
img_path = './pics/000004.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
img_path = './pics/000005.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
img_path = './pics/000121.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
img_path = './pics/000122.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
img_path = './pics/000123.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
img_path = './pics/000124.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
img_path = './pics/000125.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
tar_img = ['000001.jpg', '000002.jpg', '000003.jpg', '000004.jpg', 
           '000005.jpg', '000121.jpg', '000122.jpg', '000123.jpg', 
           '000124.jpg', '000125.jpg']

# In[]
k = 0
for pic in tar_img:
    tar_anno = gt[pic]
    
    img = images[k]
    plt.imshow(img / 255.)
    currentAxis = plt.gca()
    colors = plt.cm.hsv(np.linspace(0, 1, 11)).tolist()
    
    xmin = int(round(tar_anno[0, 0] * 300))
    ymin = int(round(tar_anno[0, 1] * 300))
    xmax = int(round(tar_anno[0, 2] * 300))
    ymax = int(round(tar_anno[0, 3] * 300))
    coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
    color = colors[k]
    k += 1
    display_txt = 'rod'
    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, 
                                        edgecolor=color, linewidth=2))
    currentAxis.text(xmin, ymin, display_txt, 
                     bbox={'facecolor':color, 'alpha':0.5})
    plt.show()